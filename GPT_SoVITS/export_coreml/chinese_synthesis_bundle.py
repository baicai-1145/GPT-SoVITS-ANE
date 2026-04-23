import argparse
import json
import os
import shutil
import sys
from typing import Dict, Optional

NOW_DIR = os.getcwd()
if NOW_DIR not in sys.path:
    sys.path.append(NOW_DIR)

from export_coreml.bundle import _write_json


def _artifact(target: str, path: str, filename: Optional[str] = None) -> Dict[str, str]:
    artifact_path = path
    return {
        "target": target,
        "filename": filename or os.path.basename(artifact_path),
        "path": artifact_path,
    }


def _ensure_exists(path: Optional[str], name: str) -> Optional[str]:
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _build_runtime_contract(
    has_g2pw: bool,
    has_prompt_bundle: bool,
    has_english_frontend: bool,
    has_yue_frontend: bool,
    has_japanese_frontend: bool,
    has_korean_frontend: bool,
) -> Dict:
    return {
        "driver": "gpt_sovits_chinese_synthesis_pipeline",
        "artifact_resolution": "prefer_artifact_path_then_filename_relative_to_bundle_dir",
        "pipeline_contract": {
            "t2s_bundle_target": "t2s_bundle",
            "vits_bundle_target": "vits_bundle",
            "zh_bert_char_target": "zh_bert_char",
            "tokenizer_target": "tokenizer",
            "optional_g2pw_target": "g2pw_bundle" if has_g2pw else None,
            "optional_prompt_target": "prompt_bundle" if has_prompt_bundle else None,
            "optional_english_frontend_target": "english_frontend_bundle" if has_english_frontend else None,
            "optional_yue_frontend_target": "yue_frontend_bundle" if has_yue_frontend else None,
            "optional_japanese_frontend_target": "japanese_frontend_bundle" if has_japanese_frontend else None,
            "optional_korean_frontend_target": "korean_frontend_bundle" if has_korean_frontend else None,
        },
    }


def _copy_or_symlink_path(source_path: str, destination_path: str, mode: str) -> None:
    parent_dir = os.path.dirname(destination_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    if os.path.lexists(destination_path):
        if os.path.isdir(destination_path) and not os.path.islink(destination_path):
            shutil.rmtree(destination_path)
        else:
            os.unlink(destination_path)

    if mode == "copy":
        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)
        return

    if mode == "symlink":
        os.symlink(os.path.abspath(source_path), destination_path)
        return

    raise ValueError(f"Unsupported materialize mode: {mode}")


def _stage_artifact(
    *,
    bundle_dir: str,
    target: str,
    source_path: str,
    materialize_mode: str,
) -> Dict[str, str]:
    if materialize_mode == "none":
        absolute_path = os.path.abspath(source_path)
        return _artifact(target, absolute_path)

    relative_path = os.path.join("artifacts", target, os.path.basename(source_path))
    staged_path = os.path.join(bundle_dir, relative_path)
    _copy_or_symlink_path(source_path, staged_path, materialize_mode)
    return _artifact(target, relative_path, filename=relative_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a top-level runtime manifest for the Apple Chinese synthesis chain."
    )
    parser.add_argument("--bundle-dir", required=True, help="Output directory containing the top-level manifest.json.")
    parser.add_argument("--t2s-bundle-dir", required=True, help="Existing T2S bundle directory.")
    parser.add_argument("--vits-bundle-dir", required=True, help="Existing VITS bundle directory.")
    parser.add_argument("--zh-bert-model", required=True, help="zh_bert_char Core ML model path.")
    parser.add_argument("--tokenizer-json", required=True, help="tokenizer.json path used by zh_bert_char.")
    parser.add_argument("--g2pw-bundle-dir", help="Optional g2pw Core ML bundle directory.")
    parser.add_argument("--prompt-bundle-dir", help="Optional prompt semantic Core ML bundle directory.")
    parser.add_argument("--english-frontend-bundle-dir", help="Optional English frontend bundle directory.")
    parser.add_argument("--yue-frontend-bundle-dir", help="Optional Yue frontend bundle directory.")
    parser.add_argument("--japanese-frontend-bundle-dir", help="Optional Japanese frontend bundle directory.")
    parser.add_argument("--korean-frontend-bundle-dir", help="Optional Korean frontend bundle directory.")
    parser.add_argument("--default-language", default="zh")
    parser.add_argument("--default-split-method", default="cut5")
    parser.add_argument(
        "--materialize-artifacts",
        choices=["copy", "symlink", "none"],
        default="copy",
        help="How the top-level bundle should stage referenced artifacts. Default: copy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.bundle_dir, exist_ok=True)

    t2s_bundle_dir = _ensure_exists(args.t2s_bundle_dir, "t2s bundle")
    vits_bundle_dir = _ensure_exists(args.vits_bundle_dir, "vits bundle")
    zh_bert_model = _ensure_exists(args.zh_bert_model, "zh_bert model")
    tokenizer_json = _ensure_exists(args.tokenizer_json, "tokenizer json")
    g2pw_bundle_dir = _ensure_exists(args.g2pw_bundle_dir, "g2pw bundle")
    prompt_bundle_dir = _ensure_exists(args.prompt_bundle_dir, "prompt bundle")
    english_frontend_bundle_dir = _ensure_exists(args.english_frontend_bundle_dir, "english frontend bundle")
    yue_frontend_bundle_dir = _ensure_exists(args.yue_frontend_bundle_dir, "yue frontend bundle")
    japanese_frontend_bundle_dir = _ensure_exists(args.japanese_frontend_bundle_dir, "japanese frontend bundle")
    korean_frontend_bundle_dir = _ensure_exists(args.korean_frontend_bundle_dir, "korean frontend bundle")

    manifest = {
        "schema_version": 1,
        "bundle_type": "gpt_sovits_chinese_synthesis_runtime_bundle",
        "bundle_dir": os.path.abspath(args.bundle_dir),
        "artifacts": {
            "t2s_bundle": _stage_artifact(
                bundle_dir=args.bundle_dir,
                target="t2s_bundle",
                source_path=t2s_bundle_dir,
                materialize_mode=args.materialize_artifacts,
            ),
            "vits_bundle": _stage_artifact(
                bundle_dir=args.bundle_dir,
                target="vits_bundle",
                source_path=vits_bundle_dir,
                materialize_mode=args.materialize_artifacts,
            ),
            "zh_bert_char": _stage_artifact(
                bundle_dir=args.bundle_dir,
                target="zh_bert_char",
                source_path=zh_bert_model,
                materialize_mode=args.materialize_artifacts,
            ),
            "tokenizer": _stage_artifact(
                bundle_dir=args.bundle_dir,
                target="tokenizer",
                source_path=tokenizer_json,
                materialize_mode=args.materialize_artifacts,
            ),
        },
        "runtime": {
            "defaults": {
                "language": args.default_language,
                "split_method": args.default_split_method,
            },
            "features": {
                "g2pw_frontend": g2pw_bundle_dir is not None,
                "prompt_semantic": prompt_bundle_dir is not None,
                "english_frontend": english_frontend_bundle_dir is not None,
                "yue_frontend": yue_frontend_bundle_dir is not None,
                "japanese_frontend": japanese_frontend_bundle_dir is not None,
                "korean_frontend": korean_frontend_bundle_dir is not None,
            },
            "materialization": {
                "mode": args.materialize_artifacts,
                "self_contained": args.materialize_artifacts != "none",
            },
            "driver_contract": _build_runtime_contract(
                has_g2pw=g2pw_bundle_dir is not None,
                has_prompt_bundle=prompt_bundle_dir is not None,
                has_english_frontend=english_frontend_bundle_dir is not None,
                has_yue_frontend=yue_frontend_bundle_dir is not None,
                has_japanese_frontend=japanese_frontend_bundle_dir is not None,
                has_korean_frontend=korean_frontend_bundle_dir is not None,
            ),
        },
    }

    if g2pw_bundle_dir is not None:
        manifest["artifacts"]["g2pw_bundle"] = _stage_artifact(
            bundle_dir=args.bundle_dir,
            target="g2pw_bundle",
            source_path=g2pw_bundle_dir,
            materialize_mode=args.materialize_artifacts,
        )
    if prompt_bundle_dir is not None:
        manifest["artifacts"]["prompt_bundle"] = _stage_artifact(
            bundle_dir=args.bundle_dir,
            target="prompt_bundle",
            source_path=prompt_bundle_dir,
            materialize_mode=args.materialize_artifacts,
        )
    if english_frontend_bundle_dir is not None:
        manifest["artifacts"]["english_frontend_bundle"] = _stage_artifact(
            bundle_dir=args.bundle_dir,
            target="english_frontend_bundle",
            source_path=english_frontend_bundle_dir,
            materialize_mode=args.materialize_artifacts,
        )
    if yue_frontend_bundle_dir is not None:
        manifest["artifacts"]["yue_frontend_bundle"] = _stage_artifact(
            bundle_dir=args.bundle_dir,
            target="yue_frontend_bundle",
            source_path=yue_frontend_bundle_dir,
            materialize_mode=args.materialize_artifacts,
        )
    if japanese_frontend_bundle_dir is not None:
        manifest["artifacts"]["japanese_frontend_bundle"] = _stage_artifact(
            bundle_dir=args.bundle_dir,
            target="japanese_frontend_bundle",
            source_path=japanese_frontend_bundle_dir,
            materialize_mode=args.materialize_artifacts,
        )
    if korean_frontend_bundle_dir is not None:
        manifest["artifacts"]["korean_frontend_bundle"] = _stage_artifact(
            bundle_dir=args.bundle_dir,
            target="korean_frontend_bundle",
            source_path=korean_frontend_bundle_dir,
            materialize_mode=args.materialize_artifacts,
        )

    manifest_path = os.path.join(args.bundle_dir, "manifest.json")
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "bundle_dir": os.path.abspath(args.bundle_dir),
                "manifest_path": os.path.abspath(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
