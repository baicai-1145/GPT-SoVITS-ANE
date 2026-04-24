import argparse
import json
import os
import sys
from typing import Dict

import numpy as np

NOW_DIR = os.getcwd()
if NOW_DIR not in sys.path:
    sys.path.append(NOW_DIR)

from export_coreml.bundle import _spec_to_schema, _write_json
from export_coreml.cli import (
    _build_cnhubert_target,
    _build_ssl_latent_target,
    _export_coreml,
)
from export_coreml.specs import get_target_spec
from process_ckpt import load_sovits_new


def _shape_range(lower_bound: int, upper_bound: int) -> Dict:
    return {
        "lower_bound": int(lower_bound),
        "upper_bound": int(max(lower_bound, upper_bound)),
    }


def _build_runtime_contract() -> Dict:
    return {
        "driver": "cnhubert_ssl_prompt_chain",
        "cnhubert_target": "cnhubert_encoder",
        "ssl_latent_target": "ssl_latent_extractor",
        "state_contract": {
            "cnhubert_outputs_used_by_ssl_latent": {
                "ssl_content": "ssl_content",
            },
        },
    }


def _build_audio_input_contract(
    raw_reference_sample_count_range: Dict,
    active_input_sample_count: int,
    active_input_sample_count_range: Dict,
    trailing_silence_sample_count: int,
) -> Dict:
    return {
        "channel_policy": "average_to_mono",
        "target_sample_rate": 16000,
        "trailing_silence_sample_count": int(trailing_silence_sample_count),
        "raw_reference_sample_count_range": dict(raw_reference_sample_count_range),
        "active_input_sample_count_range": dict(active_input_sample_count_range),
        "normalization": {
            "source": "python_prompt_semantic_raw_waveform_path",
            "do_normalize": False,
            "formula": "none",
        },
        "padding": {
            "mode": "dynamic_no_padding",
            "target_sample_count": int(active_input_sample_count),
        },
    }


def _resolve_sovits_sampling_rate(args) -> int:
    if args.prompt_source_sample_rate is not None:
        return int(args.prompt_source_sample_rate)
    dict_s2 = load_sovits_new(args.sovits_weights)
    return int(dict_s2["config"]["data"]["sampling_rate"])


def _resolve_trailing_silence_sample_count(args) -> int:
    if args.trailing_silence_sample_count is not None:
        return int(args.trailing_silence_sample_count)
    source_sample_rate = _resolve_sovits_sampling_rate(args)
    return int(round(source_sample_rate * args.trailing_silence_sec))


def _resolve_raw_reference_sample_count_range(raw_reference_sample_count: int) -> Dict:
    min_reference_seconds = 3.0
    min_reference_sample_count = int(round(16000 * min_reference_seconds))
    min_raw_reference_sample_count = min(
        max(min_reference_sample_count, 1),
        int(raw_reference_sample_count),
    )
    return _shape_range(min_raw_reference_sample_count, int(raw_reference_sample_count))


def _resolve_active_input_sample_count_range(
    raw_reference_sample_count_range: Dict,
    trailing_silence_sample_count: int,
) -> Dict:
    return _shape_range(
        int(raw_reference_sample_count_range["lower_bound"]) + int(trailing_silence_sample_count),
        int(raw_reference_sample_count_range["upper_bound"]) + int(trailing_silence_sample_count),
    )


def _resolve_dynamic_shape_ranges(
    cnhubert_wrapper,
    ssl_latent_wrapper,
    args,
    active_input_sample_count_range: Dict,
    ssl_content_shape: list[int],
    prompt_shape: list[int],
) -> Dict:
    import torch

    min_input_sample_count = int(active_input_sample_count_range["lower_bound"])
    max_ssl_frames = int(ssl_content_shape[-1])
    max_prompt_len = int(prompt_shape[-1])
    with torch.no_grad():
        min_ssl_content = cnhubert_wrapper(
            torch.zeros((1, min_input_sample_count), dtype=torch.float32, device=args.device)
        )
        min_ssl_frames = int(min_ssl_content.shape[-1])
        min_prompt = ssl_latent_wrapper(
            torch.zeros((1, int(args.ssl_channels), min_ssl_frames), dtype=torch.float32, device=args.device)
        )
        min_prompt_len = int(min_prompt.shape[-1])
    return {
        "input_sample_count_range": active_input_sample_count_range,
        "ssl_frame_range": _shape_range(min_ssl_frames, max_ssl_frames),
        "prompt_len_range": _shape_range(min_prompt_len, max_prompt_len),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a Core ML bundle and runtime manifest for prompt semantic extraction."
    )
    parser.add_argument("--bundle-dir", required=True, help="Output directory for mlpackage files and manifest.json.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--coreml-compute-units",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        default="all",
    )
    parser.add_argument(
        "--coreml-minimum-deployment-target",
        choices=["macos13", "macos14", "macos15", "ios17", "ios18"],
        default="macos15",
    )
    parser.add_argument("--coreml-compute-precision", choices=["float32", "float16"], default="float32")
    parser.add_argument("--cnhubert-base-path", default="GPT_SoVITS/pretrained_models/chinese-hubert-base")
    parser.add_argument("--sovits-weights", default="GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth")
    parser.add_argument("--sovits-version", default="v2ProPlus")
    parser.add_argument("--example-audio")
    parser.add_argument("--example-audio-sec", type=float, default=10.0)
    parser.add_argument("--ssl-channels", type=int, default=768)
    parser.add_argument("--ssl-frames", type=int, default=160)
    parser.add_argument("--trailing-silence-sec", type=float, default=0.3)
    parser.add_argument(
        "--trailing-silence-sample-count",
        type=int,
        help="Override the number of trailing 16k waveform samples appended before CN-HuBERT.",
    )
    parser.add_argument(
        "--prompt-source-sample-rate",
        type=int,
        help="Sampling rate used by the original Python prompt pipeline when resolving trailing silence samples.",
    )
    return parser.parse_args()


def main():
    import coremltools as ct

    args = parse_args()
    os.makedirs(args.bundle_dir, exist_ok=True)

    cnhubert_spec = get_target_spec("cnhubert_encoder")
    ssl_latent_spec = get_target_spec("ssl_latent_extractor")

    cnhubert_wrapper, cnhubert_inputs = _build_cnhubert_target(args)
    ssl_latent_wrapper, ssl_latent_inputs = _build_ssl_latent_target(args)

    cnhubert_filename = "cnhubert_encoder.mlpackage"
    ssl_latent_filename = "ssl_latent_extractor.mlpackage"
    cnhubert_output_path = os.path.join(args.bundle_dir, cnhubert_filename)
    ssl_latent_output_path = os.path.join(args.bundle_dir, ssl_latent_filename)

    trailing_silence_sample_count = _resolve_trailing_silence_sample_count(args)
    raw_reference_sample_count = int(cnhubert_inputs[0].shape[-1])
    raw_reference_sample_count_range = _resolve_raw_reference_sample_count_range(
        raw_reference_sample_count
    )
    active_input_sample_count = raw_reference_sample_count + int(trailing_silence_sample_count)
    active_input_sample_count_range = _resolve_active_input_sample_count_range(
        raw_reference_sample_count_range,
        trailing_silence_sample_count,
    )
    import torch

    with torch.no_grad():
        max_ssl_content = cnhubert_wrapper(
            torch.zeros(
                (1, int(active_input_sample_count_range["upper_bound"])),
                dtype=torch.float32,
                device=args.device,
            )
        )
        cnhubert_output_shape = [int(dim) for dim in max_ssl_content.shape]
        prompt_shape = [
            1,
            int(
                ssl_latent_wrapper(
                    torch.zeros(
                        (1, int(args.ssl_channels), int(cnhubert_output_shape[-1])),
                        dtype=torch.float32,
                        device=args.device,
                    )
                ).shape[-1]
            ),
        ]
    dynamic_shape_ranges = _resolve_dynamic_shape_ranges(
        cnhubert_wrapper,
        ssl_latent_wrapper,
        args,
        active_input_sample_count_range,
        cnhubert_output_shape,
        prompt_shape,
    )

    cnhubert_input_types = [
        ct.TensorType(
            name="input_values",
            shape=(
                1,
                ct.RangeDim(
                    lower_bound=int(dynamic_shape_ranges["input_sample_count_range"]["lower_bound"]),
                    upper_bound=int(dynamic_shape_ranges["input_sample_count_range"]["upper_bound"]),
                ),
            ),
            dtype=np.float32,
        )
    ]
    ssl_latent_input_types = [
        ct.TensorType(
            name="ssl_content",
            shape=(
                1,
                int(args.ssl_channels),
                ct.RangeDim(
                    lower_bound=int(dynamic_shape_ranges["ssl_frame_range"]["lower_bound"]),
                    upper_bound=int(dynamic_shape_ranges["ssl_frame_range"]["upper_bound"]),
                ),
            ),
            dtype=np.float32,
        )
    ]

    _export_coreml(
        cnhubert_output_path,
        cnhubert_spec,
        cnhubert_wrapper,
        cnhubert_inputs,
        input_types_override=cnhubert_input_types,
        compute_units=args.coreml_compute_units,
        minimum_deployment_target=args.coreml_minimum_deployment_target,
        compute_precision=args.coreml_compute_precision,
    )
    _export_coreml(
        ssl_latent_output_path,
        ssl_latent_spec,
        ssl_latent_wrapper,
        ssl_latent_inputs,
        input_types_override=ssl_latent_input_types,
        compute_units=args.coreml_compute_units,
        minimum_deployment_target=args.coreml_minimum_deployment_target,
        compute_precision=args.coreml_compute_precision,
    )

    manifest = {
        "schema_version": 1,
        "bundle_type": "gpt_sovits_prompt_semantic_coreml_bundle",
        "bundle_dir": os.path.abspath(args.bundle_dir),
        "artifacts": {
            "cnhubert_encoder": {
                "target": "cnhubert_encoder",
                "filename": cnhubert_filename,
                "path": os.path.abspath(cnhubert_output_path),
                "schema": _spec_to_schema(cnhubert_spec),
            },
            "ssl_latent_extractor": {
                "target": "ssl_latent_extractor",
                "filename": ssl_latent_filename,
                "path": os.path.abspath(ssl_latent_output_path),
                "schema": _spec_to_schema(ssl_latent_spec),
            },
        },
        "runtime": {
            "shapes": {
                "input_sample_count": active_input_sample_count,
                "ssl_content_shape": cnhubert_output_shape,
                "prompt_shape": prompt_shape,
                "prompt_len": prompt_shape[-1],
                "prompt_hop_samples": 640,
                **dynamic_shape_ranges,
            },
            "audio_input_contract": _build_audio_input_contract(
                raw_reference_sample_count_range,
                active_input_sample_count,
                active_input_sample_count_range,
                trailing_silence_sample_count,
            ),
            "capacity_contract": {
                "python_behavior": {
                    "raw_reference_audio_sample_count_range": dict(raw_reference_sample_count_range),
                    "trailing_silence_sample_count": int(trailing_silence_sample_count),
                },
                "current_export": {
                    "active_input_sample_count": int(active_input_sample_count),
                    "active_input_sample_count_range": dict(active_input_sample_count_range),
                    "prompt_len": int(prompt_shape[-1]),
                },
            },
            "coreml": {
                "compute_units": args.coreml_compute_units,
                "minimum_deployment_target": args.coreml_minimum_deployment_target,
                "compute_precision": args.coreml_compute_precision,
            },
            "driver_contract": _build_runtime_contract(),
        },
    }

    manifest_path = os.path.join(args.bundle_dir, "manifest.json")
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {"bundle_dir": os.path.abspath(args.bundle_dir), "manifest_path": os.path.abspath(manifest_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
