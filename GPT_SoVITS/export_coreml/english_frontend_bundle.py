import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

NOW_DIR = os.getcwd()
if NOW_DIR not in sys.path:
    sys.path.append(NOW_DIR)

from export_coreml.bundle import _write_json
from export_coreml.cli import (
    _resolve_coreml_compute_unit,
    _resolve_coreml_precision,
    _resolve_coreml_target,
)
from GPT_SoVITS.text.english import construct_homograph_dictionary, en_G2p, get_dict, get_namedict


class EnglishOOVPredictor(torch.nn.Module):
    def __init__(self, checkpoint_path: str, max_word_len: int, max_decode_len: int):
        super().__init__()
        weights = np.load(checkpoint_path)
        self.max_word_len = int(max_word_len)
        self.max_decode_len = int(max_decode_len)
        self.eos_id = 3

        self.register_buffer("enc_emb", torch.tensor(weights["enc_emb"], dtype=torch.float32))
        self.register_buffer("enc_w_ih", torch.tensor(weights["enc_w_ih"], dtype=torch.float32))
        self.register_buffer("enc_w_hh", torch.tensor(weights["enc_w_hh"], dtype=torch.float32))
        self.register_buffer("enc_b_ih", torch.tensor(weights["enc_b_ih"], dtype=torch.float32))
        self.register_buffer("enc_b_hh", torch.tensor(weights["enc_b_hh"], dtype=torch.float32))
        self.register_buffer("dec_emb", torch.tensor(weights["dec_emb"], dtype=torch.float32))
        self.register_buffer("dec_w_ih", torch.tensor(weights["dec_w_ih"], dtype=torch.float32))
        self.register_buffer("dec_w_hh", torch.tensor(weights["dec_w_hh"], dtype=torch.float32))
        self.register_buffer("dec_b_ih", torch.tensor(weights["dec_b_ih"], dtype=torch.float32))
        self.register_buffer("dec_b_hh", torch.tensor(weights["dec_b_hh"], dtype=torch.float32))
        self.register_buffer("fc_w", torch.tensor(weights["fc_w"], dtype=torch.float32))
        self.register_buffer("fc_b", torch.tensor(weights["fc_b"], dtype=torch.float32))

    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def _grucell(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        w_ih: torch.Tensor,
        w_hh: torch.Tensor,
        b_ih: torch.Tensor,
        b_hh: torch.Tensor,
    ) -> torch.Tensor:
        rzn_ih = torch.matmul(x, w_ih.transpose(0, 1)) + b_ih
        rzn_hh = torch.matmul(h, w_hh.transpose(0, 1)) + b_hh

        split = rzn_ih.size(-1) * 2 // 3
        rz_ih, n_ih = rzn_ih[:, :split], rzn_ih[:, split:]
        rz_hh, n_hh = rzn_hh[:, :split], rzn_hh[:, split:]

        rz = self._sigmoid(rz_ih + rz_hh)
        half = rz.size(-1) // 2
        r, z = rz[:, :half], rz[:, half:]

        n = torch.tanh(n_ih + r * n_hh)
        return (1.0 - z) * n + z * h

    def forward(self, input_ids: torch.Tensor, input_length: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.size(0)
        steps = input_ids.size(1)
        x = self.enc_emb[input_ids.to(dtype=torch.long)]
        hidden_size = self.enc_w_hh.size(1)
        h = torch.zeros((batch_size, hidden_size), dtype=torch.float32, device=input_ids.device)
        outputs = torch.zeros((batch_size, steps, hidden_size), dtype=torch.float32, device=input_ids.device)

        for step in range(steps):
            h = self._grucell(
                x[:, step, :],
                h,
                self.enc_w_ih,
                self.enc_w_hh,
                self.enc_b_ih,
                self.enc_b_hh,
            )
            outputs[:, step, :] = h

        last_index = torch.clamp(input_length.to(dtype=torch.long) - 1, min=0, max=steps - 1)
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        h = outputs[batch_indices, last_index, :]

        dec = self.dec_emb[torch.full((batch_size,), 2, dtype=torch.long, device=input_ids.device)]
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=input_ids.device)
        predictions = []

        for _ in range(self.max_decode_len):
            h = self._grucell(
                dec,
                h,
                self.dec_w_ih,
                self.dec_w_hh,
                self.dec_b_ih,
                self.dec_b_hh,
            )
            logits = torch.matmul(h, self.fc_w.transpose(0, 1)) + self.fc_b
            pred = torch.argmax(logits, dim=-1)
            pred = torch.where(
                finished,
                torch.full_like(pred, self.eos_id),
                pred,
            )
            predictions.append(pred)
            finished = torch.logical_or(finished, pred == self.eos_id)
            dec = self.dec_emb[pred.to(dtype=torch.long)]

        return torch.stack(predictions, dim=1).to(dtype=torch.int32)


@dataclass
class TensorSpec:
    name: str
    dtype: str
    shape: List[int]
    description: str


def _tensor_spec_to_dict(tensor_spec: TensorSpec) -> Dict:
    return {
        "name": tensor_spec.name,
        "dtype": tensor_spec.dtype,
        "shape": tensor_spec.shape,
        "description": tensor_spec.description,
    }


def _spec_to_schema(name: str, description: str, inputs: List[TensorSpec], outputs: List[TensorSpec]) -> Dict:
    return {
        "name": name,
        "description": description,
        "inputs": [_tensor_spec_to_dict(item) for item in inputs],
        "outputs": [_tensor_spec_to_dict(item) for item in outputs],
        "notes": [],
    }


def _g2p_en_paths() -> tuple[str, str]:
    spec = importlib.util.find_spec("g2p_en")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("g2p_en package data files not found.")
    base = spec.submodule_search_locations[0]
    return os.path.join(base, "checkpoint20.npz"), os.path.join(base, "homographs.en")


def _normalize_dict(raw: Dict[str, List[List[str]] | List[str]]) -> Dict[str, List[str]]:
    normalized = {}
    for word, pronunciations in raw.items():
        if not pronunciations:
            continue
        first = pronunciations[0] if isinstance(pronunciations[0], list) else pronunciations
        normalized[str(word)] = [str(item) for item in first]
    return normalized


def _build_runtime_assets(graphemes: List[str], phonemes: List[str]) -> Dict:
    checkpoint_path, homographs_path = _g2p_en_paths()
    _ = homographs_path
    return {
        "format_version": 1,
        "graphemes": graphemes,
        "phonemes": phonemes,
        "cmu_dict": _normalize_dict(get_dict()),
        "named_dict": _normalize_dict(get_namedict()),
        "homographs": {
            word: {
                "pron1": [str(item) for item in pron1],
                "pron2": [str(item) for item in pron2],
                "pos1": str(pos1),
            }
            for word, (pron1, pron2, pos1) in construct_homograph_dictionary().items()
        },
        "runtime_contract": {
            "word_tokenizer": r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,?!\\-]",
            "normalization_mode": "apple_english_subset_matching_python_gsv",
            "oov_predictor": "g2p_en_checkpoint20_gru",
        },
    }


def _shape_range(lower_bound: int, upper_bound: int) -> Dict:
    return {
        "lower_bound": int(lower_bound),
        "upper_bound": int(upper_bound),
    }


def _export_coreml_model(
    model: torch.nn.Module,
    output_path: str,
    max_word_len: int,
    compute_units: str,
    minimum_deployment_target: str,
    compute_precision: str,
) -> None:
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError("coremltools is not installed in the current environment.") from exc

    example_inputs = (
        torch.zeros((1, max_word_len), dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )
    traced = torch.jit.trace(model.eval(), example_inputs=example_inputs, strict=False)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, max_word_len), dtype=np.int32),
            ct.TensorType(name="input_length", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="phoneme_ids")],
        compute_units=_resolve_coreml_compute_unit(ct, compute_units),
        minimum_deployment_target=_resolve_coreml_target(ct, minimum_deployment_target),
        compute_precision=_resolve_coreml_precision(ct, compute_precision),
    )
    mlmodel.author = "OpenAI Codex"
    mlmodel.short_description = "GPT-SoVITS English OOV grapheme-to-phoneme predictor."
    mlmodel.user_defined_metadata["target"] = "english_oov_predictor"
    mlmodel.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Export an English frontend Core ML bundle.")
    parser.add_argument("--bundle-dir", required=True, help="Output directory for the frontend bundle.")
    parser.add_argument("--max-word-len", type=int, default=32)
    parser.add_argument("--max-decode-len", type=int, default=20)
    parser.add_argument("--coreml-compute-units", choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"], default="all")
    parser.add_argument(
        "--coreml-minimum-deployment-target",
        choices=["macos13", "macos14", "macos15", "ios17", "ios18"],
        default="macos15",
    )
    parser.add_argument("--coreml-compute-precision", choices=["float32", "float16"], default="float32")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.bundle_dir, exist_ok=True)

    g2p = en_G2p()
    checkpoint_path, homographs_path = _g2p_en_paths()
    _ = homographs_path
    predictor = EnglishOOVPredictor(
        checkpoint_path=checkpoint_path,
        max_word_len=args.max_word_len,
        max_decode_len=args.max_decode_len,
    )

    model_filename = "english_oov_predictor.mlpackage"
    assets_filename = "english_frontend_assets.json"
    model_output_path = os.path.join(args.bundle_dir, model_filename)
    assets_output_path = os.path.join(args.bundle_dir, assets_filename)

    _export_coreml_model(
        model=predictor,
        output_path=model_output_path,
        max_word_len=args.max_word_len,
        compute_units=args.coreml_compute_units,
        minimum_deployment_target=args.coreml_minimum_deployment_target,
        compute_precision=args.coreml_compute_precision,
    )

    runtime_assets = _build_runtime_assets(g2p.graphemes, g2p.phonemes)
    _write_json(assets_output_path, runtime_assets)

    input_specs = [
        TensorSpec("input_ids", "int32", [1, int(args.max_word_len)], "Lower-cased grapheme ids with </s> and right padding."),
        TensorSpec("input_length", "int32", [1], "Number of valid grapheme tokens including </s>."),
    ]
    output_specs = [
        TensorSpec("phoneme_ids", "int32", [1, int(args.max_decode_len)], "Greedy-decoded phoneme ids padded with EOS."),
    ]

    manifest = {
        "schema_version": 1,
        "bundle_type": "gpt_sovits_english_frontend_bundle",
        "bundle_dir": os.path.abspath(args.bundle_dir),
        "artifacts": {
            "model": {
                "target": "english_oov_predictor",
                "filename": model_filename,
                "path": os.path.abspath(model_output_path),
                "schema": _spec_to_schema(
                    "english_oov_predictor",
                    "English OOV grapheme-to-phoneme predictor.",
                    input_specs,
                    output_specs,
                ),
            },
            "runtime_assets": {
                "target": "english_frontend_assets",
                "filename": assets_filename,
                "path": os.path.abspath(assets_output_path),
            },
        },
        "runtime": {
            "shapes": {
                "max_word_len": int(args.max_word_len),
                "max_decode_len": int(args.max_decode_len),
            },
            "capacity_contract": {
                "python_behavior": {
                    "scope": "single_oov_word_predictor_only",
                    "sentence_length": "handled_by_host_text_frontend_not_by_max_word_len",
                },
                "current_export": {
                    "single_word_max_len": int(args.max_word_len),
                    "max_decode_len": int(args.max_decode_len),
                },
            },
            "coreml": {
                "compute_units": args.coreml_compute_units,
                "minimum_deployment_target": args.coreml_minimum_deployment_target,
                "compute_precision": args.coreml_compute_precision,
            },
            "tokens": {
                "decoder_start_id": 2,
                "eos_id": 3,
            },
        },
    }
    manifest_path = os.path.join(args.bundle_dir, "manifest.json")
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "bundle_dir": os.path.abspath(args.bundle_dir),
        "manifest_path": os.path.abspath(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
