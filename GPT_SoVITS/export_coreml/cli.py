import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torchaudio.functional as AF

NOW_DIR = os.getcwd()
if NOW_DIR not in sys.path:
    sys.path.append(NOW_DIR)

from export_coreml.loaders import (
    load_cnhubert_model,
    load_g2pw_model,
    load_speaker_encoder,
    load_t2s_decoder,
    load_vits_model,
    load_zh_bert_assets,
)
from export_coreml.g2pw import build_g2pw_inputs, make_g2pw_converter, torch_inputs_from_numpy
from export_coreml.specs import TARGET_SPECS, get_target_spec, list_target_specs
from export_coreml.wrappers import (
    CNHubertContentWrapper,
    G2PWProbabilityWrapper,
    PromptSemanticExtractorWrapper,
    SpeakerEncoderWrapper,
    VITSLatentSamplerWrapper,
    VITSDecodeConditionWrapper,
    VITSFlowWrapper,
    VITSMaskedWaveGeneratorWrapper,
    VITSPriorWrapper,
    VITSWaveGeneratorWrapper,
    ZhBertCharFeatureWrapper,
    T2SDecodeCoreWrapper,
    T2SDecodePrepareWrapper,
    T2SPrefillCoreWrapper,
    T2SDecodeStepWrapper,
    T2SPrefillPrepareWrapper,
    T2SPrefillWrapper,
    ZhBertPhoneFeatureWrapper,
)
from tools.audio_utils import load_audio_tensor


def _build_g2pw_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    converter = make_g2pw_converter(
        model_dir=args.g2pw_model_dir,
        model_source=args.g2pw_model_source,
        enable_non_traditional_chinese=True,
    )
    model_input = build_g2pw_inputs(
        converter,
        text=args.g2pw_example_text,
        batch_size=args.g2pw_batch_size,
        token_len=args.g2pw_token_len,
    )
    wrapper = G2PWProbabilityWrapper(load_g2pw_model(args.g2pw_model_dir, args.device)).eval().to(args.device)
    example_inputs = torch_inputs_from_numpy(model_input, device=args.device)
    return wrapper, example_inputs


def _torch_dtype_to_numpy(dtype: torch.dtype):
    mapping = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int64: np.int64,
        torch.int32: np.int32,
        torch.int16: np.int16,
        torch.bool: np.bool_,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype for Core ML export: {dtype}")
    return mapping[dtype]


def _load_audio_16k(audio_path: str = None, seconds: float = 1.0) -> torch.Tensor:
    if audio_path:
        audio, sample_rate = load_audio_tensor(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            audio = AF.resample(audio, sample_rate, 16000)
        return audio.float()
    samples = max(int(16000 * seconds), 1600)
    return torch.zeros((1, samples), dtype=torch.float32)


def _build_zh_bert_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    tokenizer, bert_model = load_zh_bert_assets(args.bert_path, args.device)
    wrapper = ZhBertPhoneFeatureWrapper(bert_model, phone_capacity=args.bert_phone_len).eval()
    if args.bert_char_len > 0:
        text = "今" * args.bert_char_len
    else:
        text = args.example_text or "今天天气不错。"
    tokenized = tokenizer(text, return_tensors="pt")
    char_count = len(text)
    word2ph = torch.ones((char_count,), dtype=torch.int32)
    example_inputs = (
        tokenized["input_ids"].to(args.device),
        tokenized["attention_mask"].to(args.device),
        tokenized["token_type_ids"].to(args.device),
        word2ph.to(args.device),
    )
    return wrapper.to(args.device), example_inputs


def _build_zh_bert_char_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    tokenizer, bert_model = load_zh_bert_assets(args.bert_path, args.device)
    wrapper = ZhBertCharFeatureWrapper(bert_model).eval()
    if args.bert_char_len > 0:
        text = "今" * args.bert_char_len
    else:
        text = args.example_text or "今天天气不错。"
    tokenized = tokenizer(text, return_tensors="pt")
    example_inputs = (
        tokenized["input_ids"].to(args.device),
        tokenized["attention_mask"].to(args.device),
        tokenized["token_type_ids"].to(args.device),
    )
    return wrapper.to(args.device), example_inputs


def _build_cnhubert_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    model = load_cnhubert_model(args.cnhubert_base_path, args.device)
    wrapper = CNHubertContentWrapper(model).eval().to(args.device)
    audio_16k = _load_audio_16k(args.example_audio, args.example_audio_sec)
    input_values = model.feature_extractor(
        audio_16k.squeeze(0).cpu().numpy(),
        return_tensors="pt",
        sampling_rate=16000,
    ).input_values.float()
    example_inputs = (input_values.to(args.device),)
    return wrapper, example_inputs


def _build_speaker_encoder_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    eres2net_dir = os.path.join(NOW_DIR, "GPT_SoVITS", "eres2net")
    if eres2net_dir not in sys.path:
        sys.path.append(eres2net_dir)
    import kaldi as Kaldi

    speaker_encoder = load_speaker_encoder(args.device, is_half=False)
    wrapper = SpeakerEncoderWrapper(speaker_encoder).eval().to(args.device)
    audio_16k = _load_audio_16k(args.example_audio, args.example_audio_sec)
    fbank = torch.stack(
        [Kaldi.fbank(wav_item.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav_item in audio_16k]
    ).float()
    example_inputs = (fbank.to(args.device),)
    return wrapper, example_inputs


def _build_speaker_encoder_input_types(example_inputs):
    import coremltools as ct

    fbank = example_inputs[0]
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


def _build_ssl_latent_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    model = load_vits_model(args.sovits_weights, device=args.device, is_half=False, version_override=args.sovits_version)
    wrapper = PromptSemanticExtractorWrapper(model).eval().to(args.device)
    example_inputs = (
        torch.zeros((1, args.ssl_channels, args.ssl_frames), dtype=torch.float32, device=args.device),
    )
    return wrapper, example_inputs


def _build_vits_decode_condition_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    model = load_vits_model(args.sovits_weights, device=args.device, is_half=False, version_override=args.sovits_version)
    wrapper = VITSDecodeConditionWrapper(model).eval().to(args.device)
    speaker_cond_dim = getattr(getattr(model, "sv_emb", None), "in_features", model.gin_channels)
    example_inputs = (
        torch.zeros((1, model.ref_enc.in_dim, args.vits_refer_frame_len), dtype=torch.float32, device=args.device),
        torch.zeros((1, speaker_cond_dim), dtype=torch.float32, device=args.device),
    )
    return wrapper, example_inputs


def _build_vits_prior_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    model = load_vits_model(args.sovits_weights, device=args.device, is_half=False, version_override=args.sovits_version)
    wrapper = VITSPriorWrapper(model).eval().to(args.device)
    ge_text_channels = getattr(getattr(model, "ge_to512", None), "out_features", model.gin_channels)
    example_inputs = (
        torch.zeros((1, 1, args.vits_code_len), dtype=torch.long, device=args.device),
        torch.zeros((1, args.vits_text_len), dtype=torch.long, device=args.device),
        torch.zeros((1, ge_text_channels, 1), dtype=torch.float32, device=args.device),
        torch.tensor([args.vits_code_len], dtype=torch.long, device=args.device),
        torch.tensor([args.vits_text_len], dtype=torch.long, device=args.device),
    )
    return wrapper, example_inputs


def _build_vits_wave_generator_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    model = load_vits_model(args.sovits_weights, device=args.device, is_half=False, version_override=args.sovits_version)
    wrapper = VITSWaveGeneratorWrapper(model).eval().to(args.device)
    latent_channels = model.enc_p.out_channels
    example_inputs = (
        torch.zeros((1, latent_channels, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.zeros((1, model.gin_channels, 1), dtype=torch.float32, device=args.device),
    )
    return wrapper, example_inputs


def _build_vits_masked_wave_generator_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    model = load_vits_model(args.sovits_weights, device=args.device, is_half=False, version_override=args.sovits_version)
    wrapper = VITSMaskedWaveGeneratorWrapper(model).eval().to(args.device)
    latent_channels = model.enc_p.out_channels
    example_inputs = (
        torch.zeros((1, latent_channels, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.ones((1, 1, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.zeros((1, model.gin_channels, 1), dtype=torch.float32, device=args.device),
    )
    return wrapper, example_inputs


def _build_vits_flow_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    model = load_vits_model(args.sovits_weights, device=args.device, is_half=False, version_override=args.sovits_version)
    wrapper = VITSFlowWrapper(model).eval().to(args.device)
    latent_channels = model.enc_p.out_channels
    example_inputs = (
        torch.zeros((1, latent_channels, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.ones((1, 1, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.zeros((1, model.gin_channels, 1), dtype=torch.float32, device=args.device),
    )
    return wrapper, example_inputs


def _build_vits_latent_sampler_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    model = load_vits_model(args.sovits_weights, device=args.device, is_half=False, version_override=args.sovits_version)
    wrapper = VITSLatentSamplerWrapper().eval().to(args.device)
    latent_channels = model.enc_p.out_channels
    example_inputs = (
        torch.zeros((1, latent_channels, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.zeros((1, latent_channels, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.zeros((1, latent_channels, args.vits_latent_len), dtype=torch.float32, device=args.device),
        torch.tensor([args.noise_scale], dtype=torch.float32, device=args.device),
    )
    return wrapper, example_inputs


def _build_t2s_prefill_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    decoder = load_t2s_decoder(args.t2s_weights, device=args.device, is_half=False)
    wrapper = T2SPrefillWrapper(decoder, max_decode_steps=args.max_decode_steps).eval().to(args.device)
    example_inputs = _build_t2s_prefill_inputs(args)
    return wrapper, example_inputs


def _build_t2s_prefill_inputs(args) -> Tuple[torch.Tensor, ...]:
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


def _build_t2s_prefill_prepare_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    decoder = load_t2s_decoder(args.t2s_weights, device=args.device, is_half=False)
    wrapper = T2SPrefillPrepareWrapper(decoder, max_decode_steps=args.max_decode_steps).eval().to(args.device)
    example_inputs = _build_t2s_prefill_inputs(args)
    return wrapper, example_inputs


def _build_t2s_prefill_core_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    decoder = load_t2s_decoder(args.t2s_weights, device=args.device, is_half=False)
    prepare_wrapper = T2SPrefillPrepareWrapper(decoder, max_decode_steps=args.max_decode_steps).eval().to(args.device)
    with torch.no_grad():
        example_inputs = tuple(t.to(device=args.device) for t in prepare_wrapper(*_build_t2s_prefill_inputs(args)))
    wrapper = T2SPrefillCoreWrapper(decoder, max_decode_steps=args.max_decode_steps).eval().to(args.device)
    return wrapper, example_inputs


def _build_t2s_decode_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    decoder = load_t2s_decoder(args.t2s_weights, device=args.device, is_half=False)
    prefill_wrapper = T2SPrefillWrapper(decoder, max_decode_steps=args.max_decode_steps).eval().to(args.device)
    _, sampled_token, _, cache_len, next_position, k_cache, v_cache = prefill_wrapper(*_build_t2s_prefill_inputs(args))
    wrapper = T2SDecodeStepWrapper(decoder).eval().to(args.device)
    example_inputs = (
        sampled_token.to(dtype=torch.int32, device=args.device),
        next_position.to(device=args.device),
        cache_len.to(device=args.device),
        k_cache.to(device=args.device),
        v_cache.to(device=args.device),
    )
    return wrapper, example_inputs


def _build_t2s_decode_prepare_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    decoder = load_t2s_decoder(args.t2s_weights, device=args.device, is_half=False)
    prefill_wrapper = T2SPrefillWrapper(decoder, max_decode_steps=args.max_decode_steps).eval().to(args.device)
    _, sampled_token, _, _, next_position, _, _ = prefill_wrapper(*_build_t2s_prefill_inputs(args))
    wrapper = T2SDecodePrepareWrapper(decoder).eval().to(args.device)
    example_inputs = (
        sampled_token.to(dtype=torch.int32, device=args.device),
        next_position.to(device=args.device),
    )
    return wrapper, example_inputs


def _build_t2s_decode_core_target(args) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    decoder = load_t2s_decoder(args.t2s_weights, device=args.device, is_half=False)
    prefill_wrapper = T2SPrefillWrapper(decoder, max_decode_steps=args.max_decode_steps).eval().to(args.device)
    _, sampled_token, _, cache_len, next_position, k_cache, v_cache = prefill_wrapper(*_build_t2s_prefill_inputs(args))
    prepare_wrapper = T2SDecodePrepareWrapper(decoder).eval().to(args.device)
    with torch.no_grad():
        xy_pos = prepare_wrapper(
            sampled_token.to(dtype=torch.int32, device=args.device),
            next_position.to(device=args.device),
        )
    wrapper = T2SDecodeCoreWrapper(decoder).eval().to(args.device)
    example_inputs = (
        xy_pos.to(device=args.device),
        next_position.to(device=args.device),
        cache_len.to(device=args.device),
        k_cache.to(device=args.device),
        v_cache.to(device=args.device),
    )
    return wrapper, example_inputs


BUILDERS = {
    "g2pw": _build_g2pw_target,
    "zh_bert_phone": _build_zh_bert_target,
    "zh_bert_char": _build_zh_bert_char_target,
    "cnhubert_encoder": _build_cnhubert_target,
    "speaker_encoder": _build_speaker_encoder_target,
    "ssl_latent_extractor": _build_ssl_latent_target,
    "vits_decode_condition": _build_vits_decode_condition_target,
    "vits_prior": _build_vits_prior_target,
    "vits_latent_sampler": _build_vits_latent_sampler_target,
    "vits_flow": _build_vits_flow_target,
    "vits_wave_generator": _build_vits_wave_generator_target,
    "vits_masked_wave_generator": _build_vits_masked_wave_generator_target,
    "t2s_prefill": _build_t2s_prefill_target,
    "t2s_prefill_prepare": _build_t2s_prefill_prepare_target,
    "t2s_prefill_core": _build_t2s_prefill_core_target,
    "t2s_decode_step": _build_t2s_decode_target,
    "t2s_decode_prepare": _build_t2s_decode_prepare_target,
    "t2s_decode_core": _build_t2s_decode_core_target,
}


def _trace_module(target_name: str, module: torch.nn.Module, example_inputs: Tuple[torch.Tensor, ...]):
    check_trace = target_name not in {
        "speaker_encoder",
        "t2s_prefill",
        "t2s_prefill_prepare",
        "t2s_prefill_core",
        "t2s_decode_step",
        "t2s_decode_prepare",
        "t2s_decode_core",
    }
    return torch.jit.trace(module, example_inputs=example_inputs, strict=False, check_trace=check_trace)


def _export_torchscript(target_name: str, output_path: str, module: torch.nn.Module, example_inputs):
    traced = _trace_module(target_name, module, example_inputs)
    traced.save(output_path)


def _to_output_dict(spec, raw_outputs):
    if isinstance(raw_outputs, torch.Tensor):
        raw_outputs = (raw_outputs,)
    elif isinstance(raw_outputs, list):
        raw_outputs = tuple(raw_outputs)
    elif not isinstance(raw_outputs, tuple):
        raise TypeError(f"Unsupported output type for target '{spec.name}': {type(raw_outputs)!r}")
    if len(raw_outputs) != len(spec.outputs):
        raise ValueError(
            f"Target '{spec.name}' expected {len(spec.outputs)} outputs, got {len(raw_outputs)} from the wrapper."
        )
    return {tensor_spec.name: value for tensor_spec, value in zip(spec.outputs, raw_outputs)}


def _resolve_coreml_compute_unit(ct, value: str):
    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }
    return mapping[value]


def _resolve_coreml_target(ct, value: str):
    mapping = {
        "macos13": ct.target.macOS13,
        "macos14": ct.target.macOS14,
        "macos15": ct.target.macOS15,
        "ios17": ct.target.iOS17,
        "ios18": ct.target.iOS18,
    }
    return mapping[value]


def _parse_fp16_skip_op_types(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _resolve_coreml_precision(ct, value: str, fp16_skip_op_types: list[str] | None = None):
    mapping = {
        "float32": ct.precision.FLOAT32,
        "float16": ct.precision.FLOAT16,
    }
    if value != "float16":
        return mapping[value]
    if not fp16_skip_op_types:
        return mapping[value]
    skipped_op_types = frozenset(fp16_skip_op_types)
    return ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in skipped_op_types)


def _export_coreml(
    output_path: str,
    spec,
    module: torch.nn.Module,
    example_inputs,
    input_types_override=None,
    output_types_override=None,
    compute_units: str = "all",
    minimum_deployment_target: str = "macos15",
    compute_precision: str = "float32",
    fp16_skip_op_types: list[str] | None = None,
):
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError("coremltools is not installed in the current environment.") from exc

    traced = _trace_module(spec.name, module, example_inputs)
    input_types = input_types_override
    if input_types is None:
        input_types = [
            ct.TensorType(
                name=tensor_spec.name,
                shape=tuple(tensor.shape),
                dtype=_torch_dtype_to_numpy(tensor.dtype),
            )
            for tensor_spec, tensor in zip(spec.inputs, example_inputs)
        ]
    output_types = output_types_override
    if output_types is None:
        output_types = [ct.TensorType(name=tensor_spec.name) for tensor_spec in spec.outputs]
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=input_types,
        outputs=output_types,
        compute_units=_resolve_coreml_compute_unit(ct, compute_units),
        minimum_deployment_target=_resolve_coreml_target(ct, minimum_deployment_target),
        compute_precision=_resolve_coreml_precision(ct, compute_precision, fp16_skip_op_types=fp16_skip_op_types),
    )
    model.author = "OpenAI Codex"
    model.short_description = spec.description
    model.user_defined_metadata["target"] = spec.name
    model.user_defined_metadata["coreml_compute_units"] = compute_units
    model.user_defined_metadata["minimum_deployment_target"] = minimum_deployment_target
    model.user_defined_metadata["compute_precision"] = compute_precision
    if fp16_skip_op_types:
        model.user_defined_metadata["fp16_skip_op_types"] = ",".join(fp16_skip_op_types)
    model.save(output_path)


def build_target(args):
    builder = BUILDERS[args.target]
    return builder(args)


def _print_target_list() -> None:
    for spec in list_target_specs():
        status = "implemented" if spec.implemented else "planned"
        coreml = "coreml-ready" if spec.coreml_ready else "coreml-pending"
        print(f"{spec.name}: {status}, {coreml}")
        print(f"  {spec.description}")


def _dump_specs_json() -> None:
    payload = {name: spec.to_dict() for name, spec in TARGET_SPECS.items()}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Export GPT-SoVITS inference submodels for Core ML / ANE.")
    parser.add_argument("--list-targets", action="store_true", help="List available export targets.")
    parser.add_argument("--dump-specs-json", action="store_true", help="Print the export target manifest as JSON.")
    parser.add_argument("--target", choices=sorted(TARGET_SPECS.keys()))
    parser.add_argument("--format", choices=["torchscript", "coreml"], default="torchscript")
    parser.add_argument("--output", help="Output file path.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--coreml-compute-units", choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"], default="all")
    parser.add_argument("--coreml-minimum-deployment-target", choices=["macos13", "macos14", "macos15", "ios17", "ios18"], default="macos15")
    parser.add_argument("--coreml-compute-precision", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--coreml-fp16-skip-op-types",
        help="Comma-separated MIL op types to keep in fp32 when --coreml-compute-precision=float16, e.g. matmul,softmax",
    )
    parser.add_argument("--bert-path", default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--g2pw-model-dir", default="GPT_SoVITS/text/G2PWModel")
    parser.add_argument("--g2pw-model-source", default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--g2pw-example-text", default="重庆火锅很好吃")
    parser.add_argument("--g2pw-batch-size", type=int, default=8)
    parser.add_argument("--g2pw-token-len", type=int, default=64)
    parser.add_argument("--cnhubert-base-path", default="GPT_SoVITS/pretrained_models/chinese-hubert-base")
    parser.add_argument("--t2s-weights", default="GPT_SoVITS/pretrained_models/s1v3.ckpt")
    parser.add_argument("--sovits-weights", default="GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth")
    parser.add_argument("--sovits-version", default="v2ProPlus")
    parser.add_argument("--example-text", default="今天天气不错。")
    parser.add_argument("--bert-char-len", type=int, default=0)
    parser.add_argument("--bert-phone-len", type=int, default=120)
    parser.add_argument("--example-audio")
    parser.add_argument("--example-audio-sec", type=float, default=10.0)
    parser.add_argument("--prompt-len", type=int, default=80)
    parser.add_argument("--ref-phone-len", type=int, default=80)
    parser.add_argument("--text-phone-len", type=int, default=120)
    parser.add_argument("--ssl-channels", type=int, default=768)
    parser.add_argument("--ssl-frames", type=int, default=200)
    parser.add_argument("--vits-refer-frame-len", type=int, default=200)
    parser.add_argument("--vits-code-len", type=int, default=40)
    parser.add_argument("--vits-text-len", type=int, default=80)
    parser.add_argument("--vits-latent-len", type=int, default=80)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--max-decode-steps", type=int, default=1500)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_targets:
        _print_target_list()
        return
    if args.dump_specs_json:
        _dump_specs_json()
        return
    if not args.target:
        raise SystemExit("--target is required unless --list-targets or --dump-specs-json is used.")

    spec = get_target_spec(args.target)
    if not spec.implemented:
        raise SystemExit(f"Target '{args.target}' is planned but not implemented yet.")
    if args.format == "coreml" and not spec.coreml_ready:
        raise SystemExit(f"Target '{args.target}' does not expose a stable Core ML export path yet.")
    if not args.output:
        raise SystemExit("--output is required for export.")

    module, example_inputs = build_target(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.format == "torchscript":
        _export_torchscript(args.target, args.output, module, example_inputs)
    else:
        input_types_override = None
        if args.target == "speaker_encoder":
            input_types_override = _build_speaker_encoder_input_types(example_inputs)
        _export_coreml(
            args.output,
            spec,
            module,
            example_inputs,
            input_types_override=input_types_override,
            compute_units=args.coreml_compute_units,
            minimum_deployment_target=args.coreml_minimum_deployment_target,
            compute_precision=args.coreml_compute_precision,
            fp16_skip_op_types=_parse_fp16_skip_op_types(args.coreml_fp16_skip_op_types),
        )
    print(f"Exported {args.target} -> {args.output}")


if __name__ == "__main__":
    main()
