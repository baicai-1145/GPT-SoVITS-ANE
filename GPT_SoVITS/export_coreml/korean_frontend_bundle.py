import argparse
import json
import os
import shutil
import sys
from typing import Dict, Optional

NOW_DIR = os.getcwd()
GPT_SOVITS_ROOT = os.path.join(NOW_DIR, "GPT_SoVITS")
for path in (NOW_DIR, GPT_SOVITS_ROOT):
    if path not in sys.path:
        sys.path.append(path)

from export_coreml.bundle import _write_json
from GPT_SoVITS.text import korean as project_korean


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


def _package_root(module) -> str:
    return os.path.dirname(os.path.abspath(module.__file__))


def _mecab_dictionary_dir() -> str:
    import mecab_ko_dic

    return os.fspath(mecab_ko_dic.dictionary_path)


def _mecab_rc_file() -> str:
    import mecab

    return os.path.abspath(os.path.join(os.path.dirname(mecab.__file__), "mecabrc"))


def _mecab_dynamic_library() -> str:
    import mecab

    return os.path.abspath(os.path.join(os.path.dirname(mecab.__file__), ".dylibs", "libmecab.2.dylib"))


def _load_g2pk2_assets() -> Dict:
    import g2pk2
    from g2pk2.utils import get_rule_id2text, parse_table

    package_root = _package_root(g2pk2)
    idioms_path = os.path.join(package_root, "idioms.txt")
    rules_path = os.path.join(package_root, "rules.txt")

    with open(idioms_path, "r", encoding="utf-8") as handle:
        idioms_lines = [line.rstrip("\n") for line in handle]
    with open(rules_path, "r", encoding="utf-8") as handle:
        rules_text = handle.read()

    return {
        "package_root": package_root,
        "table": [
            {
                "pattern": pattern,
                "replacement": replacement,
                "rule_ids": list(rule_ids),
            }
            for pattern, replacement, rule_ids in parse_table()
        ],
        "rule_id_to_text": get_rule_id2text(),
        "idioms_lines": idioms_lines,
        "rules_text": rules_text,
    }


def _load_g2pk2_effective_special_assets() -> Dict:
    return {
        "steps": [
            {
                "name": "jyeo",
                "rule_id": "5.1",
                "replacements": [
                    {"pattern": "([ᄌᄍᄎ])ᅧ", "replacement": r"\1ᅥ"},
                ],
            },
            {
                "name": "consonant_ui",
                "rule_id": "5.3",
                "replacements": [
                    {"pattern": "([ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄌᄍᄎᄏᄐᄑᄒ])ᅴ", "replacement": r"\1ᅵ"},
                ],
            },
            {
                "name": "josa_ui",
                "rule_id": "5.4.2",
                "replacements": [
                    {"pattern": "(/J)", "replacement": ""},
                ],
            },
            {
                "name": "jamo",
                "rule_id": "16",
                "replacements": [
                    {"pattern": "([그])ᆮᄋ", "replacement": r"\1ᄉ"},
                    {"pattern": "([으])[ᆽᆾᇀᇂ]ᄋ", "replacement": r"\1ᄉ"},
                    {"pattern": "([으])[ᆿ]ᄋ", "replacement": r"\1ᄀ"},
                    {"pattern": "([으])[ᇁ]ᄋ", "replacement": r"\1ᄇ"},
                ],
            },
            {
                "name": "rieulgiyeok",
                "rule_id": "11.1",
                "replacements": [
                    {"pattern": "ᆰ/P([ᄀᄁ])", "replacement": r"ᆯᄁ"},
                ],
            },
            {
                "name": "rieulbieub",
                "rule_id": "25",
                "replacements": [
                    {"pattern": "([ᆲᆴ])/Pᄀ", "replacement": r"\1ᄁ"},
                    {"pattern": "([ᆲᆴ])/Pᄃ", "replacement": r"\1ᄄ"},
                    {"pattern": "([ᆲᆴ])/Pᄉ", "replacement": r"\1ᄊ"},
                    {"pattern": "([ᆲᆴ])/Pᄌ", "replacement": r"\1ᄍ"},
                ],
            },
            {
                "name": "verb_nieun",
                "rule_id": "24",
                "replacements": [
                    {"pattern": "([ᆫᆷ])/Pᄀ", "replacement": r"\1ᄁ"},
                    {"pattern": "([ᆫᆷ])/Pᄃ", "replacement": r"\1ᄄ"},
                    {"pattern": "([ᆫᆷ])/Pᄉ", "replacement": r"\1ᄊ"},
                    {"pattern": "([ᆫᆷ])/Pᄌ", "replacement": r"\1ᄍ"},
                    {"pattern": "ᆬ/Pᄀ", "replacement": "ᆫᄁ"},
                    {"pattern": "ᆬ/Pᄃ", "replacement": "ᆫᄄ"},
                    {"pattern": "ᆬ/Pᄉ", "replacement": "ᆫᄊ"},
                    {"pattern": "ᆬ/Pᄌ", "replacement": "ᆫᄍ"},
                    {"pattern": "ᆱ/Pᄀ", "replacement": "ᆷᄁ"},
                    {"pattern": "ᆱ/Pᄃ", "replacement": "ᆷᄄ"},
                    {"pattern": "ᆱ/Pᄉ", "replacement": "ᆷᄊ"},
                    {"pattern": "ᆱ/Pᄌ", "replacement": "ᆷᄍ"},
                ],
            },
            {
                "name": "balb",
                "rule_id": "10.1",
                "replacements": [
                    {"pattern": "(바)ᆲ($|[^ᄋᄒ])", "replacement": r"\1ᆸ\2"},
                    {"pattern": "(너)ᆲ([ᄌᄍ]ᅮ|[ᄃᄄ]ᅮ)", "replacement": r"\1ᆸ\2"},
                ],
            },
            {
                "name": "palatalize",
                "rule_id": "17",
                "replacements": [
                    {"pattern": "ᆮᄋ([ᅵᅧ])", "replacement": r"ᄌ\1"},
                    {"pattern": "ᇀᄋ([ᅵᅧ])", "replacement": r"ᄎ\1"},
                    {"pattern": "ᆴᄋ([ᅵᅧ])", "replacement": r"ᆯᄎ\1"},
                    {"pattern": "ᆮᄒ([ᅵ])", "replacement": r"ᄎ\1"},
                ],
            },
            {
                "name": "modifying_rieul",
                "rule_id": "27",
                "replacements": [
                    {"pattern": "ᆯ/E ᄀ", "replacement": "ᆯ ᄁ"},
                    {"pattern": "ᆯ/E ᄃ", "replacement": "ᆯ ᄄ"},
                    {"pattern": "ᆯ/E ᄇ", "replacement": "ᆯ ᄈ"},
                    {"pattern": "ᆯ/E ᄉ", "replacement": "ᆯ ᄊ"},
                    {"pattern": "ᆯ/E ᄌ", "replacement": "ᆯ ᄍ"},
                    {"pattern": "ᆯ걸", "replacement": "ᆯ껄"},
                    {"pattern": "ᆯ밖에", "replacement": "ᆯ빠께"},
                    {"pattern": "ᆯ세라", "replacement": "ᆯ쎄라"},
                    {"pattern": "ᆯ수록", "replacement": "ᆯ쑤록"},
                    {"pattern": "ᆯ지라도", "replacement": "ᆯ찌라도"},
                    {"pattern": "ᆯ지언정", "replacement": "ᆯ찌언정"},
                    {"pattern": "ᆯ진대", "replacement": "ᆯ찐대"},
                ],
            },
        ],
    }


def _load_g2pk2_english_assets() -> Dict:
    from nltk.corpus import cmudict

    return {
        "adjust_replacements": [
            {"pattern": r"\d", "replacement": ""},
            {"pattern": " T S ", "replacement": " TS "},
            {"pattern": " D Z ", "replacement": " DZ "},
            {"pattern": " AW ER ", "replacement": " AWER "},
            {"pattern": " IH R $", "replacement": " IH ER "},
            {"pattern": " EH R $", "replacement": " EH ER "},
            {"pattern": " \\$", "replacement": ""},
        ],
        "to_choseong": {
            "B": "ᄇ",
            "CH": "ᄎ",
            "D": "ᄃ",
            "DH": "ᄃ",
            "DZ": "ᄌ",
            "F": "ᄑ",
            "G": "ᄀ",
            "HH": "ᄒ",
            "JH": "ᄌ",
            "K": "ᄏ",
            "L": "ᄅ",
            "M": "ᄆ",
            "N": "ᄂ",
            "NG": "ᄋ",
            "P": "ᄑ",
            "R": "ᄅ",
            "S": "ᄉ",
            "SH": "ᄉ",
            "T": "ᄐ",
            "TH": "ᄉ",
            "TS": "ᄎ",
            "V": "ᄇ",
            "W": "W",
            "Y": "Y",
            "Z": "ᄌ",
            "ZH": "ᄌ",
        },
        "to_jungseong": {
            "AA": "ᅡ",
            "AE": "ᅢ",
            "AH": "ᅥ",
            "AO": "ᅩ",
            "AW": "ᅡ우",
            "AWER": "ᅡ워",
            "AY": "ᅡ이",
            "EH": "ᅦ",
            "ER": "ᅥ",
            "EY": "ᅦ이",
            "IH": "ᅵ",
            "IY": "ᅵ",
            "OW": "ᅩ",
            "OY": "ᅩ이",
            "UH": "ᅮ",
            "UW": "ᅮ",
        },
        "to_jongseong": {
            "B": "ᆸ",
            "CH": "ᆾ",
            "D": "ᆮ",
            "DH": "ᆮ",
            "F": "ᇁ",
            "G": "ᆨ",
            "HH": "ᇂ",
            "JH": "ᆽ",
            "K": "ᆨ",
            "L": "ᆯ",
            "M": "ᆷ",
            "N": "ᆫ",
            "NG": "ᆼ",
            "P": "ᆸ",
            "R": "ᆯ",
            "S": "ᆺ",
            "SH": "ᆺ",
            "T": "ᆺ",
            "TH": "ᆺ",
            "V": "ᆸ",
            "W": "ᆼ",
            "Y": "ᆼ",
            "Z": "ᆽ",
            "ZH": "ᆽ",
        },
        "reconstruct_pairs": [
            {"pattern": "그W", "replacement": "ᄀW"},
            {"pattern": "흐W", "replacement": "ᄒW"},
            {"pattern": "크W", "replacement": "ᄏW"},
            {"pattern": "ᄂYᅥ", "replacement": "니어"},
            {"pattern": "ᄃYᅥ", "replacement": "디어"},
            {"pattern": "ᄅYᅥ", "replacement": "리어"},
            {"pattern": "Yᅵ", "replacement": "ᅵ"},
            {"pattern": "Yᅡ", "replacement": "ᅣ"},
            {"pattern": "Yᅢ", "replacement": "ᅤ"},
            {"pattern": "Yᅥ", "replacement": "ᅧ"},
            {"pattern": "Yᅦ", "replacement": "ᅨ"},
            {"pattern": "Yᅩ", "replacement": "ᅭ"},
            {"pattern": "Yᅮ", "replacement": "ᅲ"},
            {"pattern": "Wᅡ", "replacement": "ᅪ"},
            {"pattern": "Wᅢ", "replacement": "ᅫ"},
            {"pattern": "Wᅥ", "replacement": "ᅯ"},
            {"pattern": "Wᅩ", "replacement": "ᅯ"},
            {"pattern": "Wᅮ", "replacement": "ᅮ"},
            {"pattern": "Wᅦ", "replacement": "ᅰ"},
            {"pattern": "Wᅵ", "replacement": "ᅱ"},
            {"pattern": "ᅳᅵ", "replacement": "ᅴ"},
            {"pattern": "Y", "replacement": "ᅵ"},
            {"pattern": "W", "replacement": "ᅮ"},
        ],
        "short_vowels": ["AE", "AH", "AX", "EH", "IH", "IX", "UH"],
        "vowels": list("AEIOUY"),
        "consonants": list("BCDFGHJKLMNPQRSTVWXZ"),
        "syllable_final_or_consonants": list("$BCDFGHJKLMNPQRSTVWXZ"),
        "cmu_dict": cmudict.dict(),
    }


def _load_g2pk2_numerals_assets() -> Dict:
    return {
        "bound_nouns": project_korean._korean_classifiers.split(),
        "digits": list("0123456789"),
        "digit_names": list("영일이삼사오육칠팔구"),
        "non_zero_digits": list("123456789"),
        "non_zero_digit_names": list("일이삼사오육칠팔구"),
        "modifiers": ["한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"],
        "decimals": ["열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"],
    }


def _load_ko_pron_assets() -> Dict:
    import ko_pron
    from ko_pron.data import boundary, vowels

    return {
        "package_root": _package_root(ko_pron),
        "vowels": vowels,
        "boundary": boundary,
    }


def _regex_pairs(items) -> list[Dict[str, str]]:
    return [
        {
            "pattern": regex.pattern,
            "replacement": replacement,
        }
        for regex, replacement in items
    ]


def _project_korean_assets() -> Dict:
    return {
        "korean_classifiers": project_korean._korean_classifiers.split(),
        "hangul_divided": _regex_pairs(project_korean._hangul_divided),
        "latin_to_hangul": _regex_pairs(project_korean._latin_to_hangul),
        "ipa_to_lazy_ipa": _regex_pairs(project_korean._ipa_to_lazy_ipa),
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
            " ": "空",
        },
        "separator_contract": {
            "word": "contiguous non-separator span",
            "space": "contiguous whitespace span",
            "punct": "contiguous punctuation span",
        },
        "runtime_contract": {
            "transform_steps": [
                "latin_to_hangul",
                "g2pk2.convert_eng",
                "g2pk2.annotate",
                "g2pk2.convert_num",
                "g2pk2.special_rules",
                "g2pk2_call",
                "divide_hangul",
                "fix_g2pk2_error",
                "append_terminal_period_if_needed",
            ],
            "romanization_steps": [
                "latin_to_hangul",
                "number_to_hangul",
                "ko_pron.romanise(system=ipa)",
                "ipa_to_lazy_ipa",
            ],
        },
    }


def _mecab_probe() -> Dict:
    from g2pk2 import G2p

    g2p = G2p()
    samples = [
        "안녕하세요 OpenAI 3개 file 입니다.",
        "오늘은 GPT-SoVITS를 테스트합니다.",
    ]
    return {
        "tagger_class": type(g2p.mecab).__name__,
        "samples": [
            {
                "text": text,
                "pos": [{"surface": surface, "tag": tag} for surface, tag in g2p.mecab.pos(text)],
            }
            for text in samples
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Export a Korean frontend runtime bundle scaffold.")
    parser.add_argument("--bundle-dir", required=True, help="Output directory for the frontend bundle.")
    parser.add_argument(
        "--mecab-dictionary-dir",
        default=_mecab_dictionary_dir(),
        help="python-mecab-ko dictionary directory to stage into the bundle.",
    )
    parser.add_argument(
        "--materialize-mecab-dictionary",
        choices=["copy", "symlink", "none"],
        default="copy",
        help="How to stage mecab_ko_dic/dictionary. Default: copy.",
    )
    parser.add_argument(
        "--mecab-rc-file",
        default=_mecab_rc_file(),
        help="python-mecab-ko mecabrc file to stage into the bundle.",
    )
    parser.add_argument(
        "--materialize-mecab-rc-file",
        choices=["copy", "symlink", "none"],
        default="copy",
        help="How to stage python-mecab-ko mecabrc. Default: copy.",
    )
    parser.add_argument(
        "--mecab-dynamic-library",
        default=_mecab_dynamic_library(),
        help="python-mecab-ko bundled libmecab dynamic library to stage into the bundle.",
    )
    parser.add_argument(
        "--materialize-mecab-dynamic-library",
        choices=["copy", "symlink", "none"],
        default="copy",
        help="How to stage python-mecab-ko libmecab. Default: copy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.bundle_dir, exist_ok=True)

    mecab_dictionary_dir = os.path.abspath(args.mecab_dictionary_dir)
    mecab_rc_file = os.path.abspath(args.mecab_rc_file)
    mecab_dynamic_library = os.path.abspath(args.mecab_dynamic_library)
    if not os.path.isdir(mecab_dictionary_dir):
        raise FileNotFoundError(f"mecab dictionary directory does not exist: {mecab_dictionary_dir}")
    if not os.path.isfile(mecab_rc_file):
        raise FileNotFoundError(f"mecab rc file does not exist: {mecab_rc_file}")
    if not os.path.isfile(mecab_dynamic_library):
        raise FileNotFoundError(f"mecab dynamic library does not exist: {mecab_dynamic_library}")

    assets_filename = "korean_frontend_assets.json"
    assets_output_path = os.path.join(args.bundle_dir, assets_filename)
    runtime_assets = {
        "format_version": 2,
        "project_assets": _project_korean_assets(),
        "g2pk2_assets": _load_g2pk2_assets(),
        "g2pk2_effective_special_assets": _load_g2pk2_effective_special_assets(),
        "g2pk2_english_assets": _load_g2pk2_english_assets(),
        "g2pk2_numerals_assets": _load_g2pk2_numerals_assets(),
        "ko_pron_assets": _load_ko_pron_assets(),
        "mecab_probe": _mecab_probe(),
        "blockers": [
            "Swift runtime backend is not implemented yet.",
            "Morphological analysis still depends on reproducing python-mecab-ko compatible POS tagging.",
            "Swift runtime still needs a native g2pk2 annotate + English-to-Hangul pipeline.",
        ],
    }
    _write_json(assets_output_path, runtime_assets)

    mecab_artifact = _stage_artifact(
        bundle_dir=args.bundle_dir,
        target="mecab_dictionary",
        source_path=mecab_dictionary_dir,
        materialize_mode=args.materialize_mecab_dictionary,
    )
    mecab_rc_artifact = _stage_artifact(
        bundle_dir=args.bundle_dir,
        target="mecab_rc_file",
        source_path=mecab_rc_file,
        materialize_mode=args.materialize_mecab_rc_file,
    )
    mecab_dynamic_library_artifact = _stage_artifact(
        bundle_dir=args.bundle_dir,
        target="mecab_dynamic_library",
        source_path=mecab_dynamic_library,
        materialize_mode=args.materialize_mecab_dynamic_library,
    )

    manifest = {
        "schema_version": 1,
        "bundle_type": "gpt_sovits_korean_frontend_bundle",
        "bundle_dir": os.path.abspath(args.bundle_dir),
        "artifacts": {
            "runtime_assets": {
                "target": "korean_frontend_assets",
                "filename": assets_filename,
                "path": os.path.abspath(assets_output_path),
            },
            "mecab_dictionary": mecab_artifact,
            "mecab_rc_file": mecab_rc_artifact,
            "mecab_dynamic_library": mecab_dynamic_library_artifact,
        },
        "runtime": {
            "features": {
                "g2pk2_static_rules": True,
                "g2pk2_special_rules": True,
                "g2pk2_english_assets": True,
                "g2pk2_numerals_assets": True,
                "ko_pron_static_tables": True,
                "mecab_dictionary_staged": True,
                "mecab_rc_file_staged": True,
                "mecab_dynamic_library_staged": True,
            },
            "materialization": {
                "mecab_dictionary_mode": args.materialize_mecab_dictionary,
                "mecab_rc_file_mode": args.materialize_mecab_rc_file,
                "mecab_dynamic_library_mode": args.materialize_mecab_dynamic_library,
            },
            "next_step": "Implement Apple Korean frontend backend that reproduces mecab POS + g2pk2 annotate/English/transform contract without Python runtime.",
        },
    }
    manifest_path = os.path.join(args.bundle_dir, "manifest.json")
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "bundle_dir": os.path.abspath(args.bundle_dir),
                "manifest_path": os.path.abspath(manifest_path),
                "mecab_dictionary_dir": mecab_artifact["path"],
                "mecab_rc_file": mecab_rc_artifact["path"],
                "mecab_dynamic_library": mecab_dynamic_library_artifact["path"],
                "materialize_mecab_dictionary": args.materialize_mecab_dictionary,
                "materialize_mecab_rc_file": args.materialize_mecab_rc_file,
                "materialize_mecab_dynamic_library": args.materialize_mecab_dynamic_library,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
