import gc
import os
import sys
from typing import Optional, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

NOW_DIR = os.getcwd()
if NOW_DIR not in sys.path:
    sys.path.append(NOW_DIR)

from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from feature_extractor.cnhubert import CNHubert
from module.models import SynthesizerTrn
from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from sv import SV

from export_coreml.g2pw import load_g2pw_torch_model


def _as_device(device: str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _replace_hardtanh_with_float_bounds(module: torch.nn.Module) -> None:
    for child_name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Hardtanh):
            setattr(
                module,
                child_name,
                torch.nn.Hardtanh(
                    min_val=float(child.min_val),
                    max_val=float(child.max_val),
                    inplace=False,
                ),
            )
            continue
        _replace_hardtanh_with_float_bounds(child)


def load_zh_bert_assets(base_path: str, device: str = "cpu") -> Tuple[AutoTokenizer, AutoModelForMaskedLM]:
    runtime_device = _as_device(device)
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        base_path,
        local_files_only=True,
        output_hidden_states=True,
    ).eval()
    model = model.to(runtime_device)
    return tokenizer, model


def load_cnhubert_model(base_path: str, device: str = "cpu") -> CNHubert:
    runtime_device = _as_device(device)
    model = CNHubert(base_path).eval().to(runtime_device)
    return model


def load_speaker_encoder(device: str = "cpu", is_half: bool = False) -> SV:
    encoder = SV(_as_device(device), is_half)
    _replace_hardtanh_with_float_bounds(encoder.embedding_model)
    encoder.embedding_model = encoder.embedding_model.eval()
    return encoder


def load_t2s_decoder(weights_path: str, device: str = "cpu", is_half: bool = False):
    runtime_device = _as_device(device)
    dict_s1 = torch.load(weights_path, map_location=runtime_device, weights_only=False)
    config = dict_s1["config"]
    decoder = Text2SemanticLightningModule(
        config,
        "****",
        is_train=False,
        build_t2s_transformer=False,
        build_h_module=False,
    )
    decoder.load_inference_only_state_dict(dict_s1["weight"])
    del dict_s1
    gc.collect()
    decoder.model.release_inference_only_unused_modules()
    model = decoder.model.eval().to(runtime_device)
    if is_half and runtime_device.type != "cpu":
        model = model.half()
    return model


def load_vits_model(
    weights_path: str,
    device: str = "cpu",
    is_half: bool = False,
    version_override: Optional[str] = None,
) -> SynthesizerTrn:
    runtime_device = _as_device(device)
    _, model_version, _ = get_sovits_version_from_path_fast(weights_path)
    dict_s2 = load_sovits_new(weights_path)
    hps = dict_s2["config"]
    hps["model"]["semantic_frame_rate"] = "25hz"
    if version_override is not None:
        hps["model"]["version"] = version_override
    elif "Pro" in model_version:
        hps["model"]["version"] = model_version
    elif "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        hps["model"]["version"] = "v2"
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps["model"]["version"] = "v1"
    else:
        hps["model"]["version"] = "v2"

    model = SynthesizerTrn(
        hps["data"]["filter_length"] // 2 + 1,
        hps["train"]["segment_size"] // hps["data"]["hop_length"],
        n_speakers=hps["data"]["n_speakers"],
        **hps["model"],
    )
    model.load_state_dict(dict_s2["weight"], strict=False)
    if hasattr(model, "dec"):
        model.dec.remove_weight_norm()
    model = model.eval().to(runtime_device)
    if is_half and runtime_device.type != "cpu":
        model = model.half()
    return model


def load_g2pw_model(model_dir: str, device: str = "cpu") -> torch.nn.Module:
    runtime_device = _as_device(device)
    model = load_g2pw_torch_model(model_dir).eval().to(runtime_device)
    return model
