import argparse
import json
import os
import sys
from typing import Dict, List

import torch

NOW_DIR = os.getcwd()
if NOW_DIR not in sys.path:
    sys.path.append(NOW_DIR)

from export_coreml.cli import _export_coreml
from export_coreml.loaders import load_t2s_decoder
from export_coreml.specs import get_target_spec
from export_coreml.wrappers import (
    T2SDecodeCoreWrapper,
    T2SDecodePrepareWrapper,
    T2SDecodeStepWrapper,
    T2SPrefillCoreWrapper,
    T2SPrefillPrepareWrapper,
    T2SPrefillResidualLayerNormFP32Wrapper,
    T2SPrefillWrapper,
)


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


def _build_prefill_inputs(args):
    import torch

    return (
        torch.zeros((1, args.prompt_len), dtype=torch.int32, device=args.device),
        torch.tensor([args.prompt_len], dtype=torch.int32, device=args.device),
        torch.zeros((1, args.ref_phone_len), dtype=torch.int32, device=args.device),
        torch.tensor([args.ref_phone_len], dtype=torch.int32, device=args.device),
        torch.zeros((1, args.text_phone_len), dtype=torch.int32, device=args.device),
        torch.tensor([args.text_phone_len], dtype=torch.int32, device=args.device),
        torch.zeros((1024, args.ref_phone_len), dtype=torch.float32, device=args.device),
        torch.zeros((1024, args.text_phone_len), dtype=torch.float32, device=args.device),
    )


def _build_decode_inputs_from_prefill_outputs(prefill_outputs, device: str):
    _, sampled_token, _, cache_len, next_position, k_cache, v_cache = prefill_outputs
    return (
        sampled_token.to(dtype=torch.int32, device=device),
        next_position.to(device=device),
        cache_len.to(device=device),
        k_cache.to(device=device),
        v_cache.to(device=device),
    )


def _build_runtime_contract(max_decode_steps: int, prefill_export_mode: str, decode_export_mode: str) -> Dict:
    contract = {
        "state_contract": {
            "prefill_outputs_used_by_decode": {
                "sampled_token": "last_token",
                "cache_len": "cache_len",
                "next_position": "position_index",
                "k_cache": "k_cache",
                "v_cache": "v_cache",
            },
            "decode_outputs_used_by_next_step": {
                "sampled_token": "last_token",
                "next_cache_len": "cache_len",
                "next_position": "position_index",
                "next_k_cache": "k_cache",
                "next_v_cache": "v_cache",
            },
            "logits_output": "logits",
        },
        "sampling_contract": {
            "location": "host_runtime",
            "token_source": "host_multinomial_sample(logits)",
            "decode_input_token_shape": "[1, 1]",
            "stop_condition": "eos_reached == 1 or step_count == max_decode_steps",
            "max_decode_steps": int(max_decode_steps),
        },
    }
    if prefill_export_mode == "split_prepare_core":
        contract.update(
            {
                "driver": "t2s_prefill_prepare_core_decode_loop",
                "prefill_prepare_target": "t2s_prefill_prepare",
                "prefill_core_target": "t2s_prefill_core",
                "prefill_prepare_outputs_used_by_prefill_core": {
                    "xy_pos": "xy_pos",
                    "prompt_attn_mask": "prompt_attn_mask",
                    "active_src_len": "active_src_len",
                    "next_position": "position_seed",
                },
            }
        )
    else:
        contract.update(
            {
                "driver": "t2s_prefill_decode_loop",
                "prefill_target": "t2s_prefill",
            }
        )
    if decode_export_mode == "split_prepare_core":
        contract.update(
            {
                "decode_prepare_target": "t2s_decode_prepare",
                "decode_core_target": "t2s_decode_core",
                "decode_prepare_outputs_used_by_decode_core": {
                    "xy_pos": "xy_pos",
                },
            }
        )
    else:
        contract["decode_target"] = "t2s_decode_step"
    return contract


def _write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _parse_precision_overrides(raw_value: str | None) -> Dict[str, str]:
    if not raw_value:
        return {}
    allowed_artifacts = {
        "prefill",
        "prefill_prepare",
        "prefill_core",
        "decode_step",
        "decode_prepare",
        "decode_core",
    }
    overrides: Dict[str, str] = {}
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid precision override '{item}'. Expected artifact=precision, e.g. prefill=float32"
            )
        artifact_name, precision = [part.strip() for part in item.split("=", 1)]
        if artifact_name not in allowed_artifacts:
            raise ValueError(
                f"Unsupported T2S artifact override '{artifact_name}'. Allowed: {sorted(allowed_artifacts)}"
            )
        if precision not in {"float32", "float16"}:
            raise ValueError(f"Unsupported precision '{precision}' for artifact '{artifact_name}'.")
        overrides[artifact_name] = precision
    return overrides


def _parse_fp16_skip_op_type_overrides(raw_value: str | None) -> Dict[str, List[str]]:
    if not raw_value:
        return {}
    allowed_artifacts = {
        "prefill",
        "prefill_prepare",
        "prefill_core",
        "decode_step",
        "decode_prepare",
        "decode_core",
    }
    overrides: Dict[str, List[str]] = {}
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid fp16 skip-op override '{item}'. Expected artifact=op1+op2, e.g. prefill_core=matmul+softmax"
            )
        artifact_name, raw_ops = [part.strip() for part in item.split("=", 1)]
        if artifact_name not in allowed_artifacts:
            raise ValueError(
                f"Unsupported T2S artifact fp16 skip-op override '{artifact_name}'. Allowed: {sorted(allowed_artifacts)}"
            )
        op_types = [op.strip() for op in raw_ops.split("+") if op.strip()]
        if not op_types:
            raise ValueError(f"fp16 skip-op override for '{artifact_name}' must list at least one op type.")
        overrides[artifact_name] = op_types
    return overrides


def parse_args():
    parser = argparse.ArgumentParser(description="Export a Core ML bundle and runtime manifest for T2S Swift handoff.")
    parser.add_argument("--bundle-dir", required=True, help="Output directory for mlpackage files and manifest.json.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--coreml-compute-units", choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"], default="all")
    parser.add_argument(
        "--coreml-minimum-deployment-target",
        choices=["macos13", "macos14", "macos15", "ios17", "ios18"],
        default="macos15",
    )
    parser.add_argument("--coreml-compute-precision", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--coreml-fp16-skip-op-types",
        help="Comma-separated MIL op types to keep in fp32 when --coreml-compute-precision=float16, e.g. matmul,softmax",
    )
    parser.add_argument("--t2s-weights", default="GPT_SoVITS/pretrained_models/s1v3.ckpt")
    parser.add_argument("--prompt-len", type=int, default=80)
    parser.add_argument("--ref-phone-len", type=int, default=80)
    parser.add_argument("--text-phone-len", type=int, default=120)
    parser.add_argument("--max-decode-steps", type=int, default=1500)
    parser.add_argument(
        "--prefill-variant",
        choices=["default", "residual_layernorm_fp32"],
        default="default",
        help="Experimental T2S prefill wrapper variant.",
    )
    parser.add_argument(
        "--artifact-precision-overrides",
        help="Comma-separated artifact=precision overrides, e.g. prefill=float32,decode_step=float16",
    )
    parser.add_argument(
        "--artifact-fp16-skip-op-type-overrides",
        help="Comma-separated artifact=op1+op2 overrides for float16 exports, e.g. prefill_core=matmul+softmax+linear",
    )
    parser.add_argument(
        "--prefill-export-mode",
        choices=["monolithic", "split_prepare_core"],
        default="monolithic",
        help="Choose between the legacy monolithic prefill artifact or the experimental prepare+core split.",
    )
    parser.add_argument(
        "--decode-export-mode",
        choices=["monolithic", "split_prepare_core"],
        default="monolithic",
        help="Choose between the legacy monolithic decode artifact or the experimental prepare+core split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.bundle_dir, exist_ok=True)

    decoder = load_t2s_decoder(args.t2s_weights, device=args.device, is_half=False)
    eos_token = int(decoder.EOS)

    prefill_inputs = _build_prefill_inputs(args)
    precision_overrides = _parse_precision_overrides(args.artifact_precision_overrides)
    global_fp16_skip_op_types = [
        item.strip() for item in (args.coreml_fp16_skip_op_types or "").split(",") if item.strip()
    ]
    fp16_skip_op_type_overrides = _parse_fp16_skip_op_type_overrides(args.artifact_fp16_skip_op_type_overrides)
    artifacts = {}
    artifact_compute_precision = {}
    artifact_fp16_skip_op_types = {}

    def resolve_artifact_fp16_skip_op_types(artifact_name: str, precision: str) -> List[str]:
        if precision != "float16":
            return []
        return fp16_skip_op_type_overrides.get(artifact_name, global_fp16_skip_op_types)

    if args.prefill_export_mode == "monolithic":
        prefill_spec = get_target_spec("t2s_prefill")
        prefill_wrapper_cls = (
            T2SPrefillResidualLayerNormFP32Wrapper
            if args.prefill_variant == "residual_layernorm_fp32"
            else T2SPrefillWrapper
        )
        prefill_wrapper = prefill_wrapper_cls(decoder, max_decode_steps=args.max_decode_steps).eval().to(args.device)
        with torch.no_grad():
            prefill_outputs = prefill_wrapper(*prefill_inputs)
        decode_inputs = _build_decode_inputs_from_prefill_outputs(prefill_outputs, args.device)

        prefill_filename = "t2s_prefill.mlpackage"
        prefill_output_path = os.path.join(args.bundle_dir, prefill_filename)
        prefill_precision = precision_overrides.get("prefill", args.coreml_compute_precision)
        prefill_skip_op_types = resolve_artifact_fp16_skip_op_types("prefill", prefill_precision)
        _export_coreml(
            prefill_output_path,
            prefill_spec,
            prefill_wrapper,
            prefill_inputs,
            compute_units=args.coreml_compute_units,
            minimum_deployment_target=args.coreml_minimum_deployment_target,
            compute_precision=prefill_precision,
            fp16_skip_op_types=prefill_skip_op_types,
        )
        artifacts["prefill"] = {
            "target": "t2s_prefill",
            "filename": prefill_filename,
            "path": os.path.abspath(prefill_output_path),
            "compute_precision": prefill_precision,
            "fp16_skip_op_types": prefill_skip_op_types,
            "schema": _spec_to_schema(prefill_spec),
        }
        artifact_compute_precision["prefill"] = prefill_precision
        artifact_fp16_skip_op_types["prefill"] = prefill_skip_op_types
    else:
        if args.prefill_variant != "default":
            raise ValueError("split_prepare_core mode currently supports only --prefill-variant default.")
        prepare_spec = get_target_spec("t2s_prefill_prepare")
        core_spec = get_target_spec("t2s_prefill_core")
        prepare_wrapper = T2SPrefillPrepareWrapper(decoder, max_decode_steps=args.max_decode_steps).eval().to(args.device)
        core_wrapper = T2SPrefillCoreWrapper(decoder, max_decode_steps=args.max_decode_steps).eval().to(args.device)
        with torch.no_grad():
            prepare_outputs = prepare_wrapper(*prefill_inputs)
            prefill_outputs = core_wrapper(*prepare_outputs)
        decode_inputs = _build_decode_inputs_from_prefill_outputs(prefill_outputs, args.device)

        prepare_filename = "t2s_prefill_prepare.mlpackage"
        core_filename = "t2s_prefill_core.mlpackage"
        prepare_output_path = os.path.join(args.bundle_dir, prepare_filename)
        core_output_path = os.path.join(args.bundle_dir, core_filename)
        prepare_precision = precision_overrides.get("prefill_prepare", args.coreml_compute_precision)
        core_precision = precision_overrides.get("prefill_core", args.coreml_compute_precision)
        prepare_skip_op_types = resolve_artifact_fp16_skip_op_types("prefill_prepare", prepare_precision)
        core_skip_op_types = resolve_artifact_fp16_skip_op_types("prefill_core", core_precision)

        _export_coreml(
            prepare_output_path,
            prepare_spec,
            prepare_wrapper,
            prefill_inputs,
            compute_units=args.coreml_compute_units,
            minimum_deployment_target=args.coreml_minimum_deployment_target,
            compute_precision=prepare_precision,
            fp16_skip_op_types=prepare_skip_op_types,
        )
        _export_coreml(
            core_output_path,
            core_spec,
            core_wrapper,
            tuple(t.to(device=args.device) for t in prepare_outputs),
            compute_units=args.coreml_compute_units,
            minimum_deployment_target=args.coreml_minimum_deployment_target,
            compute_precision=core_precision,
            fp16_skip_op_types=core_skip_op_types,
        )
        artifacts["prefill_prepare"] = {
            "target": "t2s_prefill_prepare",
            "filename": prepare_filename,
            "path": os.path.abspath(prepare_output_path),
            "compute_precision": prepare_precision,
            "fp16_skip_op_types": prepare_skip_op_types,
            "schema": _spec_to_schema(prepare_spec),
        }
        artifacts["prefill_core"] = {
            "target": "t2s_prefill_core",
            "filename": core_filename,
            "path": os.path.abspath(core_output_path),
            "compute_precision": core_precision,
            "fp16_skip_op_types": core_skip_op_types,
            "schema": _spec_to_schema(core_spec),
        }
        artifact_compute_precision["prefill_prepare"] = prepare_precision
        artifact_compute_precision["prefill_core"] = core_precision
        artifact_fp16_skip_op_types["prefill_prepare"] = prepare_skip_op_types
        artifact_fp16_skip_op_types["prefill_core"] = core_skip_op_types

    if args.decode_export_mode == "monolithic":
        decode_spec = get_target_spec("t2s_decode_step")
        decode_wrapper = T2SDecodeStepWrapper(decoder).eval().to(args.device)
        decode_filename = "t2s_decode_step.mlpackage"
        decode_output_path = os.path.join(args.bundle_dir, decode_filename)
        decode_precision = precision_overrides.get("decode_step", args.coreml_compute_precision)
        decode_skip_op_types = resolve_artifact_fp16_skip_op_types("decode_step", decode_precision)
        _export_coreml(
            decode_output_path,
            decode_spec,
            decode_wrapper,
            decode_inputs,
            compute_units=args.coreml_compute_units,
            minimum_deployment_target=args.coreml_minimum_deployment_target,
            compute_precision=decode_precision,
            fp16_skip_op_types=decode_skip_op_types,
        )
        artifacts["decode_step"] = {
            "target": "t2s_decode_step",
            "filename": decode_filename,
            "path": os.path.abspath(decode_output_path),
            "compute_precision": decode_precision,
            "fp16_skip_op_types": decode_skip_op_types,
            "schema": _spec_to_schema(decode_spec),
        }
        artifact_compute_precision["decode_step"] = decode_precision
        artifact_fp16_skip_op_types["decode_step"] = decode_skip_op_types
    else:
        decode_prepare_spec = get_target_spec("t2s_decode_prepare")
        decode_core_spec = get_target_spec("t2s_decode_core")
        decode_prepare_wrapper = T2SDecodePrepareWrapper(decoder).eval().to(args.device)
        decode_core_wrapper = T2SDecodeCoreWrapper(decoder).eval().to(args.device)
        with torch.no_grad():
            decode_xy_pos = decode_prepare_wrapper(decode_inputs[0], decode_inputs[1])
        decode_prepare_filename = "t2s_decode_prepare.mlpackage"
        decode_core_filename = "t2s_decode_core.mlpackage"
        decode_prepare_output_path = os.path.join(args.bundle_dir, decode_prepare_filename)
        decode_core_output_path = os.path.join(args.bundle_dir, decode_core_filename)
        decode_prepare_precision = precision_overrides.get("decode_prepare", args.coreml_compute_precision)
        decode_core_precision = precision_overrides.get("decode_core", args.coreml_compute_precision)
        decode_prepare_skip_op_types = resolve_artifact_fp16_skip_op_types("decode_prepare", decode_prepare_precision)
        decode_core_skip_op_types = resolve_artifact_fp16_skip_op_types("decode_core", decode_core_precision)

        _export_coreml(
            decode_prepare_output_path,
            decode_prepare_spec,
            decode_prepare_wrapper,
            (decode_inputs[0], decode_inputs[1]),
            compute_units=args.coreml_compute_units,
            minimum_deployment_target=args.coreml_minimum_deployment_target,
            compute_precision=decode_prepare_precision,
            fp16_skip_op_types=decode_prepare_skip_op_types,
        )
        _export_coreml(
            decode_core_output_path,
            decode_core_spec,
            decode_core_wrapper,
            (
                decode_xy_pos.to(device=args.device),
                decode_inputs[1],
                decode_inputs[2],
                decode_inputs[3],
                decode_inputs[4],
            ),
            compute_units=args.coreml_compute_units,
            minimum_deployment_target=args.coreml_minimum_deployment_target,
            compute_precision=decode_core_precision,
            fp16_skip_op_types=decode_core_skip_op_types,
        )
        artifacts["decode_prepare"] = {
            "target": "t2s_decode_prepare",
            "filename": decode_prepare_filename,
            "path": os.path.abspath(decode_prepare_output_path),
            "compute_precision": decode_prepare_precision,
            "fp16_skip_op_types": decode_prepare_skip_op_types,
            "schema": _spec_to_schema(decode_prepare_spec),
        }
        artifacts["decode_core"] = {
            "target": "t2s_decode_core",
            "filename": decode_core_filename,
            "path": os.path.abspath(decode_core_output_path),
            "compute_precision": decode_core_precision,
            "fp16_skip_op_types": decode_core_skip_op_types,
            "schema": _spec_to_schema(decode_core_spec),
        }
        artifact_compute_precision["decode_prepare"] = decode_prepare_precision
        artifact_compute_precision["decode_core"] = decode_core_precision
        artifact_fp16_skip_op_types["decode_prepare"] = decode_prepare_skip_op_types
        artifact_fp16_skip_op_types["decode_core"] = decode_core_skip_op_types

    manifest = {
        "schema_version": 1,
        "bundle_type": "gpt_sovits_t2s_coreml_bundle",
        "bundle_dir": os.path.abspath(args.bundle_dir),
        "artifacts": artifacts,
        "runtime": {
            "eos_token": eos_token,
            "max_decode_steps": int(args.max_decode_steps),
            "sampling_defaults": {
                "mode": "host_multinomial",
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "repetition_penalty": 1.35,
            },
            "shapes": {
                "prompt_len": int(args.prompt_len),
                "ref_phone_len": int(args.ref_phone_len),
                "text_phone_len": int(args.text_phone_len),
            },
            "prefill_export_mode": args.prefill_export_mode,
            "decode_export_mode": args.decode_export_mode,
            "coreml": {
                "compute_units": args.coreml_compute_units,
                "minimum_deployment_target": args.coreml_minimum_deployment_target,
                "compute_precision": args.coreml_compute_precision,
                "artifact_compute_precision": artifact_compute_precision,
                "artifact_fp16_skip_op_types": artifact_fp16_skip_op_types,
            },
            "driver_contract": _build_runtime_contract(
                args.max_decode_steps,
                args.prefill_export_mode,
                args.decode_export_mode,
            ),
            "prefill_variant": args.prefill_variant,
        },
    }
    manifest_path = os.path.join(args.bundle_dir, "manifest.json")
    _write_json(manifest_path, manifest)
    print(json.dumps({"bundle_dir": os.path.abspath(args.bundle_dir), "manifest_path": os.path.abspath(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
