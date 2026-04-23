import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Optional

NOW_DIR = os.getcwd()
GPT_SOVITS_ROOT = os.path.join(NOW_DIR, "GPT_SoVITS")
for path in (NOW_DIR, GPT_SOVITS_ROOT):
    if path not in sys.path:
        sys.path.append(path)

from export_coreml.bundle import _write_json
from GPT_SoVITS.text import japanese as project_japanese


def _artifact(target: str, path: str, filename: Optional[str] = None) -> Dict[str, str]:
    artifact_path = path
    return {
        "target": target,
        "filename": filename or os.path.basename(artifact_path),
        "path": artifact_path,
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
        return _artifact(target, os.path.abspath(source_path))

    relative_path = os.path.join("artifacts", target, os.path.basename(source_path))
    staged_path = os.path.join(bundle_dir, relative_path)
    _copy_or_symlink_path(source_path, staged_path, materialize_mode)
    return _artifact(target, relative_path, filename=relative_path)


def _pyopenjtalk_dictionary_dir() -> str:
    import pyopenjtalk

    value = getattr(pyopenjtalk, "OPEN_JTALK_DICT_DIR")
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _default_openjtalk_source_dir() -> str:
    configured = os.environ.get("GPTSOVITS_OPENJTALK_SOURCE_DIR")
    if configured:
        return os.path.abspath(configured)
    return os.path.join(tempfile.gettempdir(), "gpt_sovits_openjtalk_src")


def _default_openjtalk_build_dir() -> str:
    configured = os.environ.get("GPTSOVITS_OPENJTALK_BUILD_DIR")
    if configured:
        return os.path.abspath(configured)
    return os.path.join(tempfile.gettempdir(), "gpt_sovits_openjtalk_build")


def _build_openjtalk_dynamic_library(source_dir: str, build_dir: str) -> str:
    if not shutil.which("cmake"):
        raise FileNotFoundError("cmake is required to build standalone open_jtalk shared library")

    source_dir = os.path.abspath(source_dir)
    build_dir = os.path.abspath(build_dir)
    if not os.path.isdir(source_dir):
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/r9y9/open_jtalk", source_dir],
            check=True,
            cwd=NOW_DIR,
        )

    subprocess.run(
        [
            "cmake",
            "-S",
            os.path.join(source_dir, "src"),
            "-B",
            build_dir,
            "-DBUILD_SHARED_LIBS=ON",
            "-DBUILD_PROGRAMS=OFF",
        ],
        check=True,
        cwd=NOW_DIR,
    )
    subprocess.run(
        ["cmake", "--build", build_dir, "-j4"],
        check=True,
        cwd=NOW_DIR,
    )
    library_path = os.path.join(build_dir, "libopenjtalk.dylib")
    if not os.path.isfile(library_path):
        raise FileNotFoundError(f"Standalone open_jtalk shared library was not produced: {library_path}")
    return library_path


def _regex_pairs(items) -> list[Dict[str, str]]:
    return [
        {
            "pattern": regex.pattern,
            "replacement": replacement,
        }
        for regex, replacement in items
    ]


def _project_japanese_assets() -> Dict:
    return {
        "japanese_characters_pattern": project_japanese._japanese_characters.pattern,
        "japanese_marks_pattern": project_japanese._japanese_marks.pattern,
        "symbols_to_japanese": _regex_pairs(project_japanese._symbols_to_japanese),
        "real_sokuon": _regex_pairs(project_japanese._real_sokuon),
        "real_hatsuon": _regex_pairs(project_japanese._real_hatsuon),
        "prosody_marks": sorted(project_japanese._prosody_marks),
        "post_replace_map": {
            "：": ",",
            "；": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": ".",
            "·": ",",
            "、": ",",
            "...": "…",
        },
        "runtime_contract": {
            "sentence_flow": [
                "symbols_to_japanese",
                "lowercase_latin",
                "split_with_marks_regex",
                "openjtalk.text2mecab",
                "openjtalk.mecab2njd",
                "openjtalk.njd_set_*",
                "openjtalk.njd2jpcommon",
                "openjtalk.jpcommon_make_label",
                "align_frontend_words_to_full_tokens",
            ],
            "with_prosody": True,
        },
    }


def _pyopenjtalk_probe() -> Dict:
    import pyopenjtalk

    samples = [
        "こんにちは。",
        "Hello.こんにちは！今日もNiCe天気ですね！",
    ]
    return {
        "dictionary_dir": _pyopenjtalk_dictionary_dir(),
        "samples": [
            {
                "text": text,
                "g2p": pyopenjtalk.g2p(text),
                "frontend_size": len(pyopenjtalk.run_frontend(text)),
            }
            for text in samples
        ],
    }


def _project_probe() -> Dict:
    samples = [
        "こんにちは。",
        "tokyotowerに行きましょう！",
    ]
    return {
        "samples": [
            {
                "text": text,
                "normalized": project_japanese.text_normalize(text),
                "phones": project_japanese.g2p(text),
                "phone_units": project_japanese.g2p_with_phone_units(text)[1],
            }
            for text in samples
        ]
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Export a Japanese frontend runtime bundle scaffold.")
    parser.add_argument("--bundle-dir", required=True, help="Output directory for the frontend bundle.")
    parser.add_argument(
        "--openjtalk-dictionary-dir",
        default=_pyopenjtalk_dictionary_dir(),
        help="pyopenjtalk dictionary directory to stage into the bundle.",
    )
    parser.add_argument(
        "--user-dictionary-dir",
        default=os.path.abspath(os.path.join(GPT_SOVITS_ROOT, "text", "ja_userdic")),
        help="GPT-SoVITS Japanese user dictionary directory to stage into the bundle.",
    )
    parser.add_argument(
        "--materialize-openjtalk-dictionary",
        choices=["copy", "symlink", "none"],
        default="copy",
        help="How to stage the OpenJTalk dictionary. Default: copy.",
    )
    parser.add_argument(
        "--materialize-user-dictionary",
        choices=["copy", "symlink", "none"],
        default="copy",
        help="How to stage the Japanese user dictionary directory. Default: copy.",
    )
    parser.add_argument(
        "--openjtalk-dynamic-library",
        help="Standalone open_jtalk shared library to stage into the bundle.",
    )
    parser.add_argument(
        "--materialize-openjtalk-dynamic-library",
        choices=["copy", "symlink", "none"],
        default="copy",
        help="How to stage the native open_jtalk dynamic library. Default: copy.",
    )
    parser.add_argument(
        "--openjtalk-source-dir",
        default=_default_openjtalk_source_dir(),
        help="Source directory for building standalone open_jtalk. If missing, the exporter clones the official repo.",
    )
    parser.add_argument(
        "--openjtalk-build-dir",
        default=_default_openjtalk_build_dir(),
        help="Build directory for standalone open_jtalk shared library.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.bundle_dir, exist_ok=True)

    openjtalk_dictionary_dir = os.path.abspath(args.openjtalk_dictionary_dir)
    user_dictionary_dir = os.path.abspath(args.user_dictionary_dir)
    openjtalk_dynamic_library = os.path.abspath(args.openjtalk_dynamic_library) if args.openjtalk_dynamic_library else _build_openjtalk_dynamic_library(
        args.openjtalk_source_dir,
        args.openjtalk_build_dir,
    )
    if not os.path.isdir(openjtalk_dictionary_dir):
        raise FileNotFoundError(f"OpenJTalk dictionary directory does not exist: {openjtalk_dictionary_dir}")
    if not os.path.isdir(user_dictionary_dir):
        raise FileNotFoundError(f"Japanese user dictionary directory does not exist: {user_dictionary_dir}")
    if not os.path.isfile(openjtalk_dynamic_library):
        raise FileNotFoundError(f"OpenJTalk dynamic library does not exist: {openjtalk_dynamic_library}")

    assets_filename = "japanese_frontend_assets.json"
    assets_output_path = os.path.join(args.bundle_dir, assets_filename)
    runtime_assets = {
        "format_version": 1,
        "project_assets": _project_japanese_assets(),
        "pyopenjtalk_probe": _pyopenjtalk_probe(),
        "project_probe": _project_probe(),
        "blockers": [],
    }
    _write_json(assets_output_path, runtime_assets)

    openjtalk_artifact = _stage_artifact(
        bundle_dir=args.bundle_dir,
        target="openjtalk_dictionary",
        source_path=openjtalk_dictionary_dir,
        materialize_mode=args.materialize_openjtalk_dictionary,
    )
    user_dictionary_artifact = _stage_artifact(
        bundle_dir=args.bundle_dir,
        target="ja_user_dictionary",
        source_path=user_dictionary_dir,
        materialize_mode=args.materialize_user_dictionary,
    )
    openjtalk_dynamic_library_artifact = _stage_artifact(
        bundle_dir=args.bundle_dir,
        target="openjtalk_dynamic_library",
        source_path=openjtalk_dynamic_library,
        materialize_mode=args.materialize_openjtalk_dynamic_library,
    )

    manifest = {
        "schema_version": 1,
        "bundle_type": "gpt_sovits_japanese_frontend_bundle",
        "bundle_dir": os.path.abspath(args.bundle_dir),
        "artifacts": {
            "runtime_assets": {
                "target": "japanese_frontend_assets",
                "filename": assets_filename,
                "path": os.path.abspath(assets_output_path),
            },
            "openjtalk_dictionary": openjtalk_artifact,
            "ja_user_dictionary": user_dictionary_artifact,
            "openjtalk_dynamic_library": openjtalk_dynamic_library_artifact,
        },
        "runtime": {
            "features": {
                "openjtalk_dictionary_staged": True,
                "ja_user_dictionary_staged": True,
                "openjtalk_dynamic_library_staged": True,
            },
            "materialization": {
                "openjtalk_dictionary_mode": args.materialize_openjtalk_dictionary,
                "ja_user_dictionary_mode": args.materialize_user_dictionary,
                "openjtalk_dynamic_library_mode": args.materialize_openjtalk_dynamic_library,
            },
            "next_step": "Run Apple native Japanese frontend parity and migrate synthesis default backend.",
        },
    }
    manifest_path = os.path.join(args.bundle_dir, "manifest.json")
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "bundle_dir": os.path.abspath(args.bundle_dir),
                "manifest_path": os.path.abspath(manifest_path),
                "openjtalk_dictionary_dir": openjtalk_artifact["path"],
                "ja_user_dictionary_dir": user_dictionary_artifact["path"],
                "openjtalk_dynamic_library": openjtalk_dynamic_library_artifact["path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
