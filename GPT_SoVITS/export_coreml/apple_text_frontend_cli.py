import argparse
import json
import os
import sys
import unicodedata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
GPT_SOVITS_ROOT = os.path.join(REPO_ROOT, "GPT_SoVITS")
OPEN_JTALK_DICT_DIR_ENV = "OPEN_JTALK_DICT_DIR"
JA_USER_DIC_DIR_ENV = "GPTSOVITS_JA_USER_DIC_DIR"
MECAB_KO_DIC_DIR_ENV = "GPTSOVITS_MECAB_KO_DIC_DIR"

for path in (REPO_ROOT, GPT_SOVITS_ROOT):
    if path not in sys.path:
        sys.path.append(path)


SUPPORTED_BASE_LANGUAGES = {"zh", "yue", "ja", "ko", "en"}


def parse_args():
    parser = argparse.ArgumentParser(description="Emit GPT-SoVITS text frontend phone payload as JSON.")
    parser.add_argument("--text", required=True, help="Input text.")
    parser.add_argument("--language", required=True, help="Frontend language.")
    parser.add_argument("--version", default="v2", help="Cleaner version.")
    return parser.parse_args()


def resolve_base_language(language: str) -> str:
    if language.startswith("all_"):
        return language[4:]
    if language in {"auto", "auto_yue"}:
        raise ValueError(f"language={language} is not supported by this segment-level CLI")
    return language


def normalize_phone_units(phone_units, version: str):
    if not phone_units:
        return []
    from GPT_SoVITS.text import cleaned_text_to_sequence

    normalized = []
    for raw_unit in phone_units:
        unit = dict(raw_unit)
        unit_phones = list(unit.get("phones", []))
        unit["phones"] = unit_phones
        unit["phone_ids"] = cleaned_text_to_sequence(unit_phones, version)
        unit["phone_start"] = int(unit.get("phone_start", 0))
        unit["phone_end"] = int(unit.get("phone_end", unit["phone_start"] + len(unit_phones)))
        unit["phone_count"] = int(unit.get("phone_count", len(unit_phones)))
        unit["char_start"] = int(unit.get("char_start", 0))
        unit["char_end"] = int(unit.get("char_end", unit["char_start"]))
        normalized.append(unit)
    return normalized


def configure_runtime_environment(base_language: str):
    if base_language == "ja":
        for key in (OPEN_JTALK_DICT_DIR_ENV, JA_USER_DIC_DIR_ENV):
            value = os.environ.get(key)
            if value:
                os.environ[key] = os.path.abspath(value)
    elif base_language == "ko":
        value = os.environ.get(MECAB_KO_DIC_DIR_ENV)
        if value:
            os.environ[MECAB_KO_DIC_DIR_ENV] = os.path.abspath(value)


def main():
    args = parse_args()
    base_language = resolve_base_language(args.language)
    if base_language not in SUPPORTED_BASE_LANGUAGES:
        raise ValueError(f"Unsupported base language: {base_language}")

    configure_runtime_environment(base_language)

    from GPT_SoVITS.text import cleaned_text_to_sequence
    from GPT_SoVITS.text.cleaner import clean_text_with_phone_units

    source_text = unicodedata.normalize("NFC", args.text)
    phones, word2ph, norm_text, phone_units = clean_text_with_phone_units(source_text, base_language, args.version)
    payload = {
        "source_text": source_text,
        "language": args.language,
        "base_language": base_language,
        "normalized_text": norm_text,
        "phones": phones,
        "phone_ids": cleaned_text_to_sequence(phones, args.version),
        "word2ph": word2ph,
        "phone_units": normalize_phone_units(phone_units, args.version),
        "backend": "python.clean_text_with_phone_units",
        "version": args.version,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
