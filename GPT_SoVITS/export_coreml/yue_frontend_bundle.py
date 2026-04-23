import argparse
import json
import os
import sys
from typing import Dict, List

NOW_DIR = os.getcwd()
if NOW_DIR not in sys.path:
    sys.path.append(NOW_DIR)

from export_coreml.bundle import _write_json
from GPT_SoVITS.text.cantonese import INITIALS, rep_map
from GPT_SoVITS.text.symbols import punctuation
from GPT_SoVITS.text.zh_normalization.char_convert import t2s_dict


def _load_tojyutping_root():
    try:
        from ToJyutping import Trie as trie_module
    except ImportError as exc:
        raise RuntimeError("ToJyutping package is not available in the current environment.") from exc
    return trie_module.root


def _serialize_value(value) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _flatten_phrase_lookup() -> Dict[str, List[str]]:
    root = _load_tojyutping_root()
    result: Dict[str, List[str]] = {}

    def walk(node, prefix: str = ""):
        if getattr(node, "v", None):
            result[prefix] = _serialize_value(node.v[0])
        for char, child in node.items():
            walk(child, prefix + char)

    walk(root)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Export a Cantonese frontend runtime bundle.")
    parser.add_argument("--bundle-dir", required=True, help="Output directory for the frontend bundle.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.bundle_dir, exist_ok=True)

    phrase_lookup = _flatten_phrase_lookup()
    assets_filename = "yue_frontend_assets.json"
    assets_output_path = os.path.join(args.bundle_dir, assets_filename)

    runtime_assets = {
        "format_version": 1,
        "max_key_length": max(len(key) for key in phrase_lookup),
        "phrase_lookup": phrase_lookup,
        "initials": list(INITIALS),
        "punctuation": list(punctuation),
        "replacement_map": dict(rep_map),
        "traditional_to_simplified_map": dict(t2s_dict),
        "digit_map": {
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        },
        "operator_map": {
            "+": "加",
            "-": "减",
            "×": "乘",
            "÷": "除",
            "=": "等",
        },
        "runtime_contract": {
            "lookup": "tojyutping_longest_prefix_lookup",
            "phrase_policy": "multi_char_phrase_expands_to_per_char_units",
            "single_char_multi_syllable_policy": "single_char_unit_keeps_multiple_syllables",
            "normalization_mode": "apple_cantonese_subset_matching_python_gsv",
        },
    }
    _write_json(assets_output_path, runtime_assets)

    manifest = {
        "schema_version": 1,
        "bundle_type": "gpt_sovits_yue_frontend_bundle",
        "bundle_dir": os.path.abspath(args.bundle_dir),
        "artifacts": {
            "runtime_assets": {
                "target": "yue_frontend_assets",
                "filename": assets_filename,
                "path": os.path.abspath(assets_output_path),
            }
        },
        "runtime": {
            "max_key_length": int(runtime_assets["max_key_length"]),
            "entry_count": int(len(phrase_lookup)),
        },
    }
    manifest_path = os.path.join(args.bundle_dir, "manifest.json")
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "bundle_dir": os.path.abspath(args.bundle_dir),
        "manifest_path": os.path.abspath(manifest_path),
        "entry_count": len(phrase_lookup),
        "max_key_length": runtime_assets["max_key_length"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
