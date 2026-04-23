from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import onnx
import torch

from text.g2pw.dataset import prepare_onnx_input
from text.g2pw.onnx_api import G2PWOnnxConverter


def load_g2pw_torch_model(model_dir: str) -> torch.nn.Module:
    try:
        from onnx2torch import convert
    except ImportError as exc:
        raise RuntimeError(
            "onnx2torch is required to rebuild g2pw into an exportable PyTorch graph. "
            "Install it with `pip install onnx2torch`."
        ) from exc

    onnx_path = _resolve_g2pw_onnx_path(model_dir)
    return convert(onnx.load(onnx_path)).eval()


def make_g2pw_converter(
    model_dir: str,
    model_source: str,
    enable_non_traditional_chinese: bool = True,
) -> G2PWOnnxConverter:
    return G2PWOnnxConverter(
        model_dir=model_dir,
        style="pinyin",
        model_source=model_source,
        enable_non_tradional_chinese=enable_non_traditional_chinese,
    )


def build_g2pw_inputs(
    converter: G2PWOnnxConverter,
    text: str,
    batch_size: int,
    token_len: int,
) -> Dict[str, np.ndarray]:
    texts, model_query_ids, *_ = converter._prepare_data([text])
    model_input = prepare_onnx_input(
        tokenizer=converter.tokenizer,
        labels=converter.labels,
        char2phonemes=converter.char2phonemes,
        chars=converter.chars,
        texts=texts,
        query_ids=model_query_ids,
        use_mask=converter.config.use_mask,
        window_size=None,
        char2id=converter.char2id,
        char_phoneme_masks=converter.char_phoneme_masks,
    )
    if not model_input:
        raise ValueError(f"g2pw example text produced no model inputs: {text!r}")
    return pad_g2pw_inputs(model_input, batch_size=batch_size, token_len=token_len)


def pad_g2pw_inputs(
    model_input: Dict[str, np.ndarray],
    batch_size: int,
    token_len: int,
) -> Dict[str, np.ndarray]:
    actual_batch = int(model_input["input_ids"].shape[0])
    actual_token_len = int(model_input["input_ids"].shape[1])
    if actual_batch <= 0:
        raise ValueError("g2pw padding requires at least one valid query row")
    if actual_batch > batch_size:
        raise ValueError(f"g2pw query count {actual_batch} exceeds batch capacity {batch_size}")
    if actual_token_len > token_len:
        raise ValueError(f"g2pw token length {actual_token_len} exceeds token capacity {token_len}")

    label_count = int(model_input["phoneme_masks"].shape[1])
    padded = {
        "input_ids": np.zeros((batch_size, token_len), dtype=np.int64),
        "token_type_ids": np.zeros((batch_size, token_len), dtype=np.int64),
        "attention_masks": np.zeros((batch_size, token_len), dtype=np.int64),
        "phoneme_masks": np.zeros((batch_size, label_count), dtype=np.float32),
        "char_ids": np.zeros((batch_size,), dtype=np.int64),
        "position_ids": np.zeros((batch_size,), dtype=np.int64),
    }

    padded["input_ids"][:actual_batch, :actual_token_len] = model_input["input_ids"]
    padded["token_type_ids"][:actual_batch, :actual_token_len] = model_input["token_type_ids"]
    padded["attention_masks"][:actual_batch, :actual_token_len] = model_input["attention_masks"]
    padded["phoneme_masks"][:actual_batch, :] = model_input["phoneme_masks"]
    padded["char_ids"][:actual_batch] = model_input["char_ids"]
    padded["position_ids"][:actual_batch] = model_input["position_ids"]

    if actual_batch < batch_size:
        last_valid_index = actual_batch - 1
        # g2pw ends with a masked softmax; leaving padded rows as all-zero masks would
        # create 0/0 NaNs. Duplicate the last valid query so fixed-capacity export stays
        # numerically stable while callers ignore rows beyond the real query count.
        padded["input_ids"][actual_batch:, :] = padded["input_ids"][last_valid_index]
        padded["token_type_ids"][actual_batch:, :] = padded["token_type_ids"][last_valid_index]
        padded["attention_masks"][actual_batch:, :] = padded["attention_masks"][last_valid_index]
        padded["phoneme_masks"][actual_batch:, :] = padded["phoneme_masks"][last_valid_index]
        padded["char_ids"][actual_batch:] = padded["char_ids"][last_valid_index]
        padded["position_ids"][actual_batch:] = padded["position_ids"][last_valid_index]
    return padded


def torch_inputs_from_numpy(model_input: Dict[str, np.ndarray], device: str = "cpu") -> Tuple[torch.Tensor, ...]:
    runtime_device = torch.device(device)
    return (
        torch.from_numpy(model_input["input_ids"]).to(runtime_device),
        torch.from_numpy(model_input["token_type_ids"]).to(runtime_device),
        torch.from_numpy(model_input["attention_masks"]).to(runtime_device),
        torch.from_numpy(model_input["phoneme_masks"]).to(runtime_device),
        torch.from_numpy(model_input["char_ids"]).to(runtime_device),
        torch.from_numpy(model_input["position_ids"]).to(runtime_device),
    )


def _resolve_g2pw_onnx_path(model_dir: str) -> str:
    candidates = [
        os.path.join(model_dir, "g2pW.onnx"),
        os.path.join(model_dir, "g2pw.onnx"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Cannot locate g2pw ONNX model in {model_dir}. Tried: {candidates}")
