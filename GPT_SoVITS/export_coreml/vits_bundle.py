import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import torch

NOW_DIR = os.getcwd()
if NOW_DIR not in sys.path:
    sys.path.append(NOW_DIR)

from export_coreml.cli import _export_coreml
from export_coreml.loaders import load_speaker_encoder, load_vits_model
from export_coreml.specs import get_target_spec
from export_coreml.wrappers import (
    SpeakerEncoderWrapper,
    VITSDecodeConditionWrapper,
    VITSFlowWrapper,
    VITSLatentSamplerWrapper,
    VITSMaskedWaveGeneratorWrapper,
    VITSPriorWrapper,
    VITSWaveGeneratorWrapper,
)
from process_ckpt import load_sovits_new


def _tensor_spec_to_dict(tensor_spec) -> Dict:
    return {
        "name": tensor_spec.name,
        "dtype": tensor_spec.dtype,
        "shape": tensor_spec.shape,
        "description": tensor_spec.description,
    }


def _spec_to_schema(spec) -> Dict:
    return {
        "name": spec.name,
        "description": spec.description,
        "inputs": [_tensor_spec_to_dict(item) for item in spec.inputs],
        "outputs": [_tensor_spec_to_dict(item) for item in spec.outputs],
        "notes": list(spec.notes),
    }


def _write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _parse_precision_overrides(raw_value: str | None, allowed_artifacts: set[str]) -> Dict[str, str]:
    if not raw_value:
        return {}
    overrides: Dict[str, str] = {}
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid precision override '{item}'. Expected artifact=precision, e.g. decode_condition=float32"
            )
        artifact_name, precision = [part.strip() for part in item.split("=", 1)]
        if artifact_name not in allowed_artifacts:
            raise ValueError(
                f"Unsupported VITS artifact override '{artifact_name}'. Allowed: {sorted(allowed_artifacts)}"
            )
        if precision not in {"float32", "float16"}:
            raise ValueError(f"Unsupported precision '{precision}' for artifact '{artifact_name}'.")
        overrides[artifact_name] = precision
    return overrides


def _shape_range(lower_bound: int, upper_bound: int) -> Dict:
    return {
        "lower_bound": int(lower_bound),
        "upper_bound": int(upper_bound),
    }


def _resolve_input_types_override(shape_mode: str, builder, *builder_args):
    if shape_mode == "fixed":
        return None
    return builder(*builder_args)


def _build_decode_condition_inputs(model, args):
    speaker_cond_dim = getattr(getattr(model, "sv_emb", None), "in_features", model.gin_channels)
    return (
        torch.zeros((1, model.ref_enc.in_dim, args.vits_refer_frame_len), dtype=torch.float32, device=args.device),
        torch.zeros((1, speaker_cond_dim), dtype=torch.float32, device=args.device),
    )


def _build_decode_condition_input_types(model, args):
    import coremltools as ct

    speaker_cond_dim = getattr(getattr(model, "sv_emb", None), "in_features", model.gin_channels)
    return [
        ct.TensorType(
            name="refer",
            shape=(
                1,
                int(model.ref_enc.in_dim),
                ct.RangeDim(
                    lower_bound=1,
                    upper_bound=int(args.vits_refer_frame_len),
                ),
            ),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="sv_emb",
            shape=(1, int(speaker_cond_dim)),
            dtype=np.float32,
        ),
    ]


def _build_prior_inputs(model, args):
    ge_text_channels = getattr(getattr(model, "ge_to512", None), "out_features", model.gin_channels)
    return (
        torch.zeros((1, 1, args.vits_code_len), dtype=torch.long, device=args.device),
        torch.zeros((1, args.vits_text_len), dtype=torch.long, device=args.device),
        torch.zeros((1, ge_text_channels, 1), dtype=torch.float32, device=args.device),
        torch.tensor([args.vits_code_len], dtype=torch.long, device=args.device),
        torch.tensor([args.vits_text_len], dtype=torch.long, device=args.device),
    )


def _build_prior_input_types(model, args):
    import coremltools as ct

    ge_text_channels = getattr(getattr(model, "ge_to512", None), "out_features", model.gin_channels)
    semantic_frames = ct.RangeDim(lower_bound=1, upper_bound=int(args.vits_code_len), symbol="semantic_frames")
    text_phone_count = ct.RangeDim(lower_bound=1, upper_bound=int(args.vits_text_len), symbol="text_phone_count")
    return [
        ct.TensorType(name="codes", shape=(1, 1, semantic_frames), dtype=np.int32),
        ct.TensorType(name="text", shape=(1, text_phone_count), dtype=np.int32),
        ct.TensorType(name="ge_text", shape=(1, int(ge_text_channels), 1), dtype=np.float32),
        ct.TensorType(name="code_lengths", shape=(1,), dtype=np.int32),
        ct.TensorType(name="text_lengths", shape=(1,), dtype=np.int32),
    ]


def _build_flow_inputs(model, args):
    latent_channels = model.enc_p.out_channels
    return (
        torch.zeros((1, latent_channels, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.ones((1, 1, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.zeros((1, model.gin_channels, 1), dtype=torch.float32, device=args.device),
    )


def _build_latent_like_input_types(model, args, latent_name: str):
    import coremltools as ct

    latent_channels = int(model.enc_p.out_channels)
    latent_frames = ct.RangeDim(lower_bound=1, upper_bound=int(args.vits_latent_len), symbol="latent_frames")
    return [
        ct.TensorType(name=latent_name, shape=(1, latent_channels, latent_frames), dtype=np.float32),
        ct.TensorType(name="y_mask", shape=(1, 1, latent_frames), dtype=np.float32),
        ct.TensorType(name="ge", shape=(1, int(model.gin_channels), 1), dtype=np.float32),
    ]


def _build_wave_inputs(model, args):
    latent_channels = model.enc_p.out_channels
    return (
        torch.zeros((1, latent_channels, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.zeros((1, model.gin_channels, 1), dtype=torch.float32, device=args.device),
    )


def _build_masked_wave_inputs(model, args):
    latent_channels = model.enc_p.out_channels
    return (
        torch.zeros((1, latent_channels, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.ones((1, 1, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.zeros((1, model.gin_channels, 1), dtype=torch.float32, device=args.device),
    )


def _build_latent_sampler_inputs(model, args):
    latent_channels = model.enc_p.out_channels
    return (
        torch.zeros((1, latent_channels, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.zeros((1, latent_channels, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.zeros((1, latent_channels, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.tensor([args.noise_scale], dtype=torch.float32, device=args.device),
    )


def _build_latent_sampler_input_types(model, args):
    import coremltools as ct

    latent_channels = int(model.enc_p.out_channels)
    latent_frames = ct.RangeDim(lower_bound=1, upper_bound=int(args.vits_latent_len), symbol="latent_frames")
    return [
        ct.TensorType(name="prior_mean", shape=(1, latent_channels, latent_frames), dtype=np.float32),
        ct.TensorType(name="prior_log_scale", shape=(1, latent_channels, latent_frames), dtype=np.float32),
        ct.TensorType(name="noise", shape=(1, latent_channels, latent_frames), dtype=np.float32),
        ct.TensorType(name="noise_scale", shape=(1,), dtype=np.float32),
    ]


def _build_speaker_encoder_target(args):
    eres2net_dir = os.path.join(NOW_DIR, "GPT_SoVITS", "eres2net")
    if eres2net_dir not in sys.path:
        sys.path.append(eres2net_dir)
    import kaldi as Kaldi

    speaker_encoder = load_speaker_encoder(args.device, is_half=False)
    wrapper = SpeakerEncoderWrapper(speaker_encoder).eval().to(args.device)
    samples = max(int(16000 * args.speaker_audio_sec), 1600)
    audio_16k = torch.zeros((1, samples), dtype=torch.float32, device=args.device)
    fbank = torch.stack(
        [Kaldi.fbank(wav_item.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav_item in audio_16k]
    ).float()
    return wrapper, (fbank.to(args.device),)


def _build_speaker_encoder_input_types(speaker_inputs):
    import coremltools as ct

    fbank = speaker_inputs[0]
    return [
        ct.TensorType(
            name="fbank_80",
            shape=(
                1,
                ct.RangeDim(
                    lower_bound=1,
                    upper_bound=int(fbank.shape[1]),
                ),
                int(fbank.shape[2]),
            ),
            dtype=np.float32,
        )
    ]


def _build_runtime_contract(noise_scale: float, include_speaker_encoder: bool) -> Dict:
    contract = {
        "driver": "vits_decode_chain",
        "decode_condition_target": "vits_decode_condition",
        "prior_target": "vits_prior",
        "latent_sampler_target": "vits_latent_sampler",
        "flow_target": "vits_flow",
        "wave_generator_target": "vits_masked_wave_generator",
        "state_contract": {
            "decode_condition_outputs_used_by_prior": {
                "ge_text": "ge_text",
            },
            "decode_condition_outputs_used_by_flow": {
                "ge": "ge",
            },
            "decode_condition_outputs_used_by_wave_generator": {
                "ge": "ge",
            },
            "prior_outputs_used_by_sampling": {
                "prior_mean": "prior_mean",
                "prior_log_scale": "prior_log_scale",
            },
            "sampling_output_used_by_flow": {
                "z_p": "z_p",
            },
            "prior_outputs_used_by_wave_generator": {
                "y_mask": "y_mask",
            },
            "flow_outputs_used_by_wave_generator": {
                "z": "z",
            },
        },
        "sampling_contract": {
            "location": "vits_latent_sampler",
            "formula": "z_p = prior_mean + noise * exp(prior_log_scale) * noise_scale",
            "noise_distribution": "standard_normal",
            "noise_scale": float(noise_scale),
            "noise_input_name": "noise",
            "noise_scale_input_name": "noise_scale",
        },
        "post_flow_contract": {
            "location": "vits_masked_wave_generator",
            "formula": "masked_z = z * y_mask",
        },
    }
    if include_speaker_encoder:
        contract["speaker_conditioning_contract"] = {
            "speaker_encoder_target": "speaker_encoder",
            "speaker_encoder_output_used_by_decode_condition": {
                "sv_emb": "sv_emb",
            },
        }
    return contract


def _load_vits_data_config(weights_path: str) -> Dict:
    bundle = load_sovits_new(weights_path)
    return dict(bundle["config"]["data"])


def _build_reference_audio_contract(
    vits_data_config: Dict,
    speaker_fbank_shape,
    refer_channels: int,
    refer_frame_len: int,
    refer_frame_count_range: Dict,
    shape_mode: str,
) -> Dict:
    speaker_fbank_frame_count_range = (
        _shape_range(1, int(speaker_fbank_shape[1])) if shape_mode == "dynamic" else None
    )
    return {
        "shared_waveform_contract": {
            "channel_policy": "average_stereo_to_mono",
            "amplitude_normalization": {
                "condition": "max_abs > 1.0",
                "formula": "audio /= min(2.0, max_abs)",
            },
        },
        "refer_spectrogram_contract": {
            "target_sample_rate": int(vits_data_config["sampling_rate"]),
            "spectrogram_type": "magnitude_stft",
            "n_fft": int(vits_data_config["filter_length"]),
            "hop_length": int(vits_data_config["hop_length"]),
            "win_length": int(vits_data_config["win_length"]),
            "center": False,
            "pad_mode": "reflect",
            "onesided": True,
            "magnitude_epsilon": 1e-8,
            "expected_frequency_bins": int(vits_data_config["filter_length"] // 2 + 1),
        },
        "refer_model_input_contract": {
            "source": "refer_spectrogram",
            "channel_slice_start": 0,
            "channel_slice_end": int(refer_channels),
            "target_frame_length": int(refer_frame_len),
            "frame_count_range": dict(refer_frame_count_range) if shape_mode == "dynamic" else None,
            "frame_padding": (
                "dynamic_no_padding_or_right_truncate"
                if shape_mode == "dynamic"
                else "fixed_right_pad_or_right_truncate"
            ),
        },
        "speaker_fbank_80_contract": {
            "source": "normalized_reference_waveform_resampled_from_32k_to_16k",
            "target_sample_rate": 16000,
            "feature_type": "kaldi_fbank",
            "num_mel_bins": 80,
            "dither": 0.0,
            "frame_count_range": speaker_fbank_frame_count_range,
            "example_input_shape": list(speaker_fbank_shape),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Export a Core ML bundle and runtime manifest for the split VITS decode chain.")
    parser.add_argument("--bundle-dir", required=True, help="Output directory for mlpackage files and manifest.json.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--coreml-compute-units", choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"], default="all")
    parser.add_argument(
        "--coreml-minimum-deployment-target",
        choices=["macos13", "macos14", "macos15", "ios17", "ios18"],
        default="macos15",
    )
    parser.add_argument("--coreml-compute-precision", choices=["float32", "float16"], default="float32")
    parser.add_argument("--sovits-weights", default="GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth")
    parser.add_argument("--sovits-version", default="v2ProPlus")
    parser.add_argument("--vits-refer-frame-len", type=int, default=200)
    parser.add_argument("--vits-code-len", type=int, default=40)
    parser.add_argument("--vits-text-len", type=int, default=80)
    parser.add_argument("--vits-latent-len", type=int, default=80)
    parser.add_argument("--speaker-audio-sec", type=float, default=10.0)
    parser.add_argument(
        "--shape-mode",
        choices=["dynamic", "fixed"],
        default="dynamic",
        help="Export bounded dynamic RangeDim inputs or fixed-capacity inputs.",
    )
    parser.add_argument("--exclude-speaker-encoder", action="store_true")
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument(
        "--artifact-precision-overrides",
        help="Comma-separated artifact=precision overrides, e.g. decode_condition=float32,speaker_encoder=float32",
    )
    return parser.parse_args()


def export_vits_bundle(args):
    expected_latent_len = int(args.vits_code_len) * 2
    if int(args.vits_latent_len) != expected_latent_len:
        raise ValueError(
            f"vits_latent_len mismatch: expected {expected_latent_len} from vits_code_len={args.vits_code_len}, "
            f"got {args.vits_latent_len}"
        )

    os.makedirs(args.bundle_dir, exist_ok=True)
    model = load_vits_model(args.sovits_weights, device=args.device, is_half=False, version_override=args.sovits_version)
    vits_data_config = _load_vits_data_config(args.sovits_weights)
    include_speaker_encoder = not bool(args.exclude_speaker_encoder)
    allowed_artifacts = {"decode_condition", "prior", "latent_sampler", "flow", "wave_generator", "speaker_encoder"}
    precision_overrides = _parse_precision_overrides(args.artifact_precision_overrides, allowed_artifacts)

    targets = {
        "decode_condition": (
            "vits_decode_condition",
            "vits_decode_condition.mlpackage",
            VITSDecodeConditionWrapper(model).eval().to(args.device),
            _build_decode_condition_inputs(model, args),
        ),
        "prior": (
            "vits_prior",
            "vits_prior.mlpackage",
            VITSPriorWrapper(model).eval().to(args.device),
            _build_prior_inputs(model, args),
        ),
        "latent_sampler": (
            "vits_latent_sampler",
            "vits_latent_sampler.mlpackage",
            VITSLatentSamplerWrapper().eval().to(args.device),
            _build_latent_sampler_inputs(model, args),
        ),
        "flow": (
            "vits_flow",
            "vits_flow.mlpackage",
            VITSFlowWrapper(model).eval().to(args.device),
            _build_flow_inputs(model, args),
        ),
        "wave_generator": (
            "vits_masked_wave_generator",
            "vits_masked_wave_generator.mlpackage",
            VITSMaskedWaveGeneratorWrapper(model).eval().to(args.device),
            _build_masked_wave_inputs(model, args),
        ),
    }
    speaker_fbank_shape = None
    if include_speaker_encoder:
        speaker_wrapper, speaker_inputs = _build_speaker_encoder_target(args)
        speaker_fbank_shape = tuple(int(dim) for dim in speaker_inputs[0].shape)
        targets["speaker_encoder"] = (
            "speaker_encoder",
            "speaker_encoder.mlpackage",
            speaker_wrapper,
            speaker_inputs,
        )

    artifacts = {}
    for artifact_name, (target_name, filename, wrapper, example_inputs) in targets.items():
        spec = get_target_spec(target_name)
        output_path = os.path.join(args.bundle_dir, filename)
        artifact_precision = precision_overrides.get(artifact_name, args.coreml_compute_precision)
        input_types_override = None
        if artifact_name == "decode_condition":
            input_types_override = _resolve_input_types_override(
                args.shape_mode,
                _build_decode_condition_input_types,
                model,
                args,
            )
        elif artifact_name == "prior":
            input_types_override = _resolve_input_types_override(
                args.shape_mode,
                _build_prior_input_types,
                model,
                args,
            )
        elif artifact_name == "latent_sampler":
            input_types_override = _resolve_input_types_override(
                args.shape_mode,
                _build_latent_sampler_input_types,
                model,
                args,
            )
        elif artifact_name == "flow":
            input_types_override = _resolve_input_types_override(
                args.shape_mode,
                _build_latent_like_input_types,
                model,
                args,
                "z_p",
            )
        elif artifact_name == "wave_generator":
            input_types_override = _resolve_input_types_override(
                args.shape_mode,
                _build_latent_like_input_types,
                model,
                args,
                "z",
            )
        elif artifact_name == "speaker_encoder":
            input_types_override = _resolve_input_types_override(
                args.shape_mode,
                _build_speaker_encoder_input_types,
                example_inputs,
            )
        _export_coreml(
            output_path,
            spec,
            wrapper,
            example_inputs,
            input_types_override=input_types_override,
            compute_units=args.coreml_compute_units,
            minimum_deployment_target=args.coreml_minimum_deployment_target,
            compute_precision=artifact_precision,
        )
        artifacts[artifact_name] = {
            "target": target_name,
            "filename": filename,
            "path": os.path.abspath(output_path),
            "compute_precision": artifact_precision,
            "schema": _spec_to_schema(spec),
        }

    manifest = {
        "schema_version": 1,
        "bundle_type": "gpt_sovits_vits_coreml_bundle",
        "bundle_dir": os.path.abspath(args.bundle_dir),
        "artifacts": artifacts,
        "runtime": {
            "shapes": {
                "refer_frame_len": int(args.vits_refer_frame_len),
                "refer_frame_count_range": _shape_range(1, int(args.vits_refer_frame_len))
                if args.shape_mode == "dynamic"
                else None,
                "semantic_code_len": int(args.vits_code_len),
                "semantic_code_len_range": _shape_range(1, int(args.vits_code_len))
                if args.shape_mode == "dynamic"
                else None,
                "text_phone_len": int(args.vits_text_len),
                "text_phone_len_range": _shape_range(1, int(args.vits_text_len))
                if args.shape_mode == "dynamic"
                else None,
                "latent_len": int(args.vits_latent_len),
                "latent_len_range": _shape_range(1, int(args.vits_latent_len))
                if args.shape_mode == "dynamic"
                else None,
            },
            "capacity_contract": {
                "python_behavior": {
                    "refer": "dynamic_reference_spectrogram_frames",
                    "codes": "dynamic_code_lengths",
                    "text": "dynamic_text_lengths",
                },
                "current_export": {
                    "refer": {
                        "shape_mode": args.shape_mode,
                        "frame_count_range": _shape_range(1, int(args.vits_refer_frame_len))
                        if args.shape_mode == "dynamic"
                        else None,
                        "fixed_capacity": int(args.vits_refer_frame_len) if args.shape_mode == "fixed" else None,
                    },
                    "codes": {
                        "shape_mode": args.shape_mode,
                        "dynamic_range": _shape_range(1, int(args.vits_code_len)) if args.shape_mode == "dynamic" else None,
                        "fixed_capacity": int(args.vits_code_len) if args.shape_mode == "fixed" else None,
                        "caller_must_supply_code_lengths": True,
                    },
                    "text": {
                        "shape_mode": args.shape_mode,
                        "dynamic_range": _shape_range(1, int(args.vits_text_len)) if args.shape_mode == "dynamic" else None,
                        "fixed_capacity": int(args.vits_text_len) if args.shape_mode == "fixed" else None,
                        "caller_must_supply_text_lengths": True,
                    },
                    "latent": {
                        "shape_mode": args.shape_mode,
                        "dynamic_range": _shape_range(1, int(args.vits_latent_len))
                        if args.shape_mode == "dynamic"
                        else None,
                        "fixed_capacity": int(args.vits_latent_len) if args.shape_mode == "fixed" else None,
                    },
                },
            },
            "coreml": {
                "compute_units": args.coreml_compute_units,
                "minimum_deployment_target": args.coreml_minimum_deployment_target,
                "compute_precision": args.coreml_compute_precision,
                "shape_mode": args.shape_mode,
                "artifact_compute_precision": {
                    artifact_name: artifacts[artifact_name]["compute_precision"] for artifact_name in artifacts
                },
            },
            "driver_contract": _build_runtime_contract(args.noise_scale, include_speaker_encoder),
        },
    }
    if speaker_fbank_shape is not None:
        refer_frame_count_range = manifest["runtime"]["shapes"]["refer_frame_count_range"]
        manifest["runtime"]["reference_audio_contract"] = _build_reference_audio_contract(
            vits_data_config,
            speaker_fbank_shape=speaker_fbank_shape,
            refer_channels=int(model.ref_enc.in_dim),
            refer_frame_len=int(args.vits_refer_frame_len),
            refer_frame_count_range=refer_frame_count_range,
            shape_mode=args.shape_mode,
        )
    manifest_path = os.path.join(args.bundle_dir, "manifest.json")
    _write_json(manifest_path, manifest)
    payload = {
        "bundle_dir": os.path.abspath(args.bundle_dir),
        "manifest_path": os.path.abspath(manifest_path),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main():
    args = parse_args()
    export_vits_bundle(args)


if __name__ == "__main__":
    main()
