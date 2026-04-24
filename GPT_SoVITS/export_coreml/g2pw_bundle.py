import argparse
import hashlib
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

import jieba_fast as jieba
import numpy as np
from pypinyin import Style, lazy_pinyin

NOW_DIR = os.getcwd()
if NOW_DIR not in sys.path:
    sys.path.append(NOW_DIR)

from export_coreml.bundle import _spec_to_schema, _write_json
from export_coreml.cli import _build_g2pw_target, _export_coreml
from export_coreml.g2pw import make_g2pw_converter
from export_coreml.specs import get_target_spec
from GPT_SoVITS.text.cleaner import clean_text_with_phone_units
from GPT_SoVITS.text import jieba_posseg_fast as psg
from GPT_SoVITS.text.chinese2 import must_erhua, not_erhua
from GPT_SoVITS.text.g2pw.pronunciation import get_phrase_pronunciation, phrase_override_dict, pp_dict
from GPT_SoVITS.text.tone_sandhi import ToneSandhi
from GPT_SoVITS.text.zh_normalization.char_convert import t2s_dict

_MANUAL_PHONE_TEMPLATE_OVERRIDES: Dict[str, Tuple[str, str]] = {
    "ㄈㄨㄥ": ("f", "eng"),
    "ㄈㄧㄠ": ("f", "iao"),
    "ㄉㄧㄤ": ("d", "iang"),
    "ㄧㄞ": ("y", "ai"),
    "ㄌㄩㄢ": ("l", "uan"),
    "ㄌㄩㄣ": ("l", "un"),
    "ㄝ": ("EE", "e"),
    "ㄩㄤ": ("y", "ang"),
}

_MARKED_VOWEL_TO_ASCII = {
    "ā": "a",
    "á": "a",
    "ǎ": "a",
    "à": "a",
    "ē": "e",
    "é": "e",
    "ě": "e",
    "è": "e",
    "ê": "e",
    "ī": "i",
    "í": "i",
    "ǐ": "i",
    "ì": "i",
    "ō": "o",
    "ó": "o",
    "ǒ": "o",
    "ò": "o",
    "ū": "u",
    "ú": "u",
    "ǔ": "u",
    "ù": "u",
    "ü": "v",
    "ǖ": "v",
    "ǘ": "v",
    "ǚ": "v",
    "ǜ": "v",
}

_AUTHORITATIVE_PROGRESS_VERSION = 2


def _shape_range(lower_bound: int, upper_bound: int) -> Dict[str, int]:
    return {
        "lower_bound": int(lower_bound),
        "upper_bound": int(upper_bound),
    }


def _load_pinyin_to_phone_map() -> Dict[str, List[str]]:
    mapping = {}
    with open(os.path.join(NOW_DIR, "GPT_SoVITS/text/opencpop-strict.txt"), "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            items = line.split("\t")
            if len(items) != 2:
                continue
            mapping[items[0]] = items[1].split(" ")
    return mapping


def _encode_posseg_state(state: Tuple[str, str]) -> str:
    return f"{state[0]}:{state[1]}"


def _build_posseg_hmm_assets() -> Dict:
    return {
        "char_state_tab": {
            char: [_encode_posseg_state(state) for state in states]
            for char, states in psg.char_state_tab_P.items()
        },
        "start_prob": {
            _encode_posseg_state(state): float(prob)
            for state, prob in psg.start_P.items()
        },
        "trans_prob": {
            _encode_posseg_state(state): {
                _encode_posseg_state(next_state): float(next_prob)
                for next_state, next_prob in next_states.items()
            }
            for state, next_states in psg.trans_P.items()
        },
        "emit_prob": {
            _encode_posseg_state(state): {
                char: float(prob)
                for char, prob in emissions.items()
            }
            for state, emissions in psg.emit_P.items()
        },
    }


def _split_tone_label(label: str) -> Optional[Tuple[str, int]]:
    if not label or label[-1] not in "12345":
        return None
    return label[:-1], int(label[-1])


def _normalize_latin_base(value: str) -> str:
    normalized = "".join(_MARKED_VOWEL_TO_ASCII.get(char, char) for char in value.lower())
    return "".join(char for char in normalized if "a" <= char <= "z")


def _make_phone_template(label: str, initial_phone: str, final_phone_base: str, tone: int) -> Dict:
    return {
        "label": label,
        "initial_phone": initial_phone,
        "final_phone_base": final_phone_base,
        "tone": int(tone),
    }


def _resolve_label_phone_template(label: str, converter, pinyin_to_phone_map: Dict[str, List[str]]) -> Optional[Dict]:
    parsed = _split_tone_label(label)
    if parsed is None:
        return None
    base, tone = parsed

    if base in _MANUAL_PHONE_TEMPLATE_OVERRIDES:
        initial_phone, final_phone_base = _MANUAL_PHONE_TEMPLATE_OVERRIDES[base]
        return _make_phone_template(label, initial_phone, final_phone_base, tone)

    latin_base = _normalize_latin_base(base)
    if latin_base in pinyin_to_phone_map:
        initial_phone, final_phone_base = pinyin_to_phone_map[latin_base]
        return _make_phone_template(label, initial_phone, final_phone_base, tone)

    converted_base = converter.bopomofo_convert_dict.get(base)
    if converted_base in pinyin_to_phone_map:
        initial_phone, final_phone_base = pinyin_to_phone_map[converted_base]
        return _make_phone_template(label, initial_phone, final_phone_base, tone)

    return None
def _build_runtime_contract(
    polyphonic_context_chars: int,
    batch_size: int,
    token_len: int,
    shape_mode: str,
) -> Dict:
    return {
        "driver": "g2pw_polyphonic_disambiguation",
        "query_alignment": "char_aligned",
        "padding_contract": {
            "batch_padding_mode": (
                "per_batch_right_pad_tokens_only_for_dynamic_bundle"
                if shape_mode == "dynamic"
                else "right_pad_rows_and_tokens_to_fixed_capacity"
            ),
            "caller_must_ignore_rows_beyond_real_query_count": False,
        },
        "tokenization_contract": {
            "tokenizer": "bert_wordpiece",
            "special_tokens": ["[CLS]", "[SEP]", "[PAD]", "[UNK]"],
            "position_id_offset": 1,
            "python_reference_max_token_len": 512,
        },
        "polyphonic_context_contract": {
            "mode": "crop_to_polyphonic_span_with_context",
            "context_chars_per_side": int(polyphonic_context_chars),
            "fallback_when_exceeds_token_len": "truncate_around_query_token",
        },
        "capacity_contract": {
            "python_behavior": {
                "query_count": "host_side_dynamic_batching",
                "token_len": 512,
            },
            "current_export": {
                "query_count": {
                    "shape_mode": shape_mode,
                    "dynamic_range": _shape_range(1, int(batch_size)) if shape_mode == "dynamic" else None,
                    "fixed_capacity": int(batch_size) if shape_mode == "fixed" else None,
                },
                "token_len": {
                    "shape_mode": shape_mode,
                    "dynamic_range": _shape_range(1, int(token_len)) if shape_mode == "dynamic" else None,
                    "fixed_capacity": int(token_len) if shape_mode == "fixed" else None,
                },
            },
        },
    }


def _build_g2pw_input_types(batch_size: int, token_len: int, label_count: int):
    import coremltools as ct

    batch = ct.RangeDim(lower_bound=1, upper_bound=int(batch_size), symbol="batch")
    tokens = ct.RangeDim(lower_bound=1, upper_bound=int(token_len), symbol="token_len")
    return [
        ct.TensorType(name="input_ids", shape=(batch, tokens), dtype=np.int32),
        ct.TensorType(name="token_type_ids", shape=(batch, tokens), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(batch, tokens), dtype=np.int32),
        ct.TensorType(name="phoneme_mask", shape=(batch, int(label_count)), dtype=np.float32),
        ct.TensorType(name="char_ids", shape=(batch,), dtype=np.int32),
        ct.TensorType(name="position_ids", shape=(batch,), dtype=np.int32),
    ]


def _resolve_input_types_override(shape_mode: str, builder, *builder_args):
    if shape_mode == "fixed":
        return None
    return builder(*builder_args)


def _contains_cjk(value: str) -> bool:
    return any(
        0x3400 <= ord(char) <= 0x4DBF or 0x4E00 <= ord(char) <= 0x9FFF
        for char in value
    )


def _contains_polyphonic_char(word: str, converter) -> bool:
    return any(char in converter.polyphonic_chars_new for char in word)


def _contains_sandhi_trigger(word: str) -> bool:
    trigger_chars = set("一不儿们子上下里来去吧呢哈啊呐噻嘛吖嗨哦哒额滴哩哟喽啰耶喔诶的地得个")
    return any(char in trigger_chars for char in word)


def _make_authoritative_phrase_entry(word: str) -> Optional[Dict[str, Any]]:
    try:
        phones, word2ph, _norm_text, phone_units = clean_text_with_phone_units(f"{word}.", "zh", "v2")
    except Exception:
        return None
    if len(word2ph) != len(word) + 1 or sum(word2ph) != len(phones):
        return None
    word2ph = word2ph[: len(word)]
    phones = phones[: sum(word2ph)]

    templates = []
    cursor = 0
    for index, phone_count in enumerate(word2ph):
        syllable_phones = phones[cursor : cursor + phone_count]
        cursor += phone_count
        if len(syllable_phones) != 2:
            return None
        initial_phone, final_phone = syllable_phones
        if not final_phone or final_phone[-1] not in "12345":
            return None
        tone = int(final_phone[-1])
        final_phone_base = final_phone[:-1]
        templates.append(
            _make_phone_template(
                label=f"{word[index]}:{initial_phone}:{final_phone}",
                initial_phone=initial_phone,
                final_phone_base=final_phone_base,
                tone=tone,
            )
        )
    authoritative_units = []
    for unit in phone_units:
        char_start = int(unit.get("char_start", 0))
        char_end = int(unit.get("char_end", 0))
        if char_end > len(word) + 1:
            return None
        authoritative_units.append(
            {
                "text": str(unit.get("text", "")),
                "pos": str(unit.get("pos", "x") or "x"),
                "char_start": char_start,
                "char_end": char_end,
            }
        )
    core_text = "".join(
        item["text"]
        for item in authoritative_units
        if item["char_start"] < len(word)
    )
    if core_text != word:
        return None
    return {
        "templates": templates,
        "unit_breakdown": authoritative_units,
    }


def _collect_authoritative_phrase_words(
    converter,
    word_frequency: Dict[str, int],
    forced_words: set[str],
    limit: Optional[int] = None,
) -> List[str]:
    explicit_words = {
        word
        for word in set(phrase_override_dict.keys()).union(pp_dict.keys())
        if len(word) > 1 and _contains_cjk(word)
    }
    candidates = set(explicit_words)
    candidates.update(
        word
        for word in word_frequency
        if len(word) > 1 and (
            _contains_polyphonic_char(word, converter)
            or _contains_sandhi_trigger(word)
            or word in forced_words
        )
    )
    candidates.update(
        word
        for word in forced_words
        if len(word) > 1 and _contains_cjk(word)
    )
    ordered = sorted(candidates)
    if limit is not None and limit > 0:
        return ordered[:limit]
    return ordered


def _authoritative_words_signature(words: List[str]) -> str:
    digest = hashlib.sha256()
    digest.update(f"v{_AUTHORITATIVE_PROGRESS_VERSION}\n".encode("utf-8"))
    for word in words:
        digest.update(word.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_progress_chunk(path: str, expected_signature: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("signature") != expected_signature:
        return {}
    if isinstance(payload, dict) and "entries" in payload:
        return {
            str(word): entry
            for word, entry in payload["entries"].items()
        }
    return {}


def _write_progress_chunk(
    *,
    path: str,
    chunk_index: int,
    word_start: int,
    word_end: int,
    entries: Dict[str, Dict[str, Any]],
    signature: str,
) -> None:
    _write_json(
        path,
        {
            "chunk_index": int(chunk_index),
            "word_start": int(word_start),
            "word_end": int(word_end),
            "signature": signature,
            "entry_count": len(entries),
            "entries": entries,
        },
    )


def _write_progress_manifest(
    *,
    progress_dir: str,
    signature: str,
    chunk_size: int,
    total_word_count: int,
) -> None:
    _write_json(
        os.path.join(progress_dir, "manifest.json"),
        {
            "signature": signature,
            "chunk_size": int(chunk_size),
            "total_word_count": int(total_word_count),
        },
    )


def _default_authoritative_progress_dir(bundle_dir: str) -> str:
    return os.path.join(f"{os.path.abspath(bundle_dir)}.build", "phrase_template_progress")


def _resolve_authoritative_progress_dir(
    *,
    bundle_dir: str,
    assets_dir: str,
    explicit_progress_dir: Optional[str],
) -> str:
    if explicit_progress_dir:
        return os.path.abspath(explicit_progress_dir)

    legacy_progress_dir = os.path.join(assets_dir, "phrase_template_progress")
    default_progress_dir = _default_authoritative_progress_dir(bundle_dir)
    if os.path.isdir(legacy_progress_dir) and not os.path.exists(default_progress_dir):
        os.makedirs(os.path.dirname(default_progress_dir), exist_ok=True)
        shutil.move(legacy_progress_dir, default_progress_dir)
        print(
            "[g2pw_bundle] moved legacy authoritative progress cache out of bundle: "
            f"{legacy_progress_dir} -> {default_progress_dir}"
        )
    return default_progress_dir


def _cleanup_bundle_local_progress_dir(*, assets_dir: str, active_progress_dir: Optional[str]) -> None:
    legacy_progress_dir = os.path.abspath(os.path.join(assets_dir, "phrase_template_progress"))
    if not os.path.isdir(legacy_progress_dir):
        return
    if active_progress_dir and os.path.abspath(active_progress_dir) == legacy_progress_dir:
        return
    shutil.rmtree(legacy_progress_dir)
    print(f"[g2pw_bundle] removed export-only bundle cache at {legacy_progress_dir}")


def _build_phrase_phone_templates(
    converter,
    authoritative_words: List[str],
    progress_dir: Optional[str] = None,
    chunk_size: int = 2000,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict[str, Any]]]]:
    phrase_templates = {}
    phrase_unit_breakdowns = {}
    if not authoritative_words:
        return phrase_templates, phrase_unit_breakdowns

    if progress_dir:
        os.makedirs(progress_dir, exist_ok=True)
        signature = _authoritative_words_signature(authoritative_words)
        _write_progress_manifest(
            progress_dir=progress_dir,
            signature=signature,
            chunk_size=chunk_size,
            total_word_count=len(authoritative_words),
        )
    else:
        signature = ""

    chunk_size = max(int(chunk_size), 1)
    total_chunks = (len(authoritative_words) + chunk_size - 1) // chunk_size
    for chunk_index, word_start in enumerate(range(0, len(authoritative_words), chunk_size)):
        word_end = min(word_start + chunk_size, len(authoritative_words))
        chunk_words = authoritative_words[word_start:word_end]
        chunk_templates: Dict[str, List[Dict]]
        chunk_breakdowns: Dict[str, List[Dict[str, Any]]]
        chunk_path = None
        if progress_dir:
            chunk_path = os.path.join(
                progress_dir,
                f"chunk_{chunk_index:05d}_{word_start:07d}_{word_end - 1:07d}.json",
            )
            if os.path.exists(chunk_path):
                chunk_entries = _load_progress_chunk(chunk_path, signature)
                if not chunk_entries:
                    try:
                        os.remove(chunk_path)
                    except OSError:
                        pass
                else:
                    chunk_templates = {
                        word: entry["templates"]
                        for word, entry in chunk_entries.items()
                        if entry.get("templates")
                    }
                    chunk_breakdowns = {
                        word: entry["unit_breakdown"]
                        for word, entry in chunk_entries.items()
                        if entry.get("unit_breakdown")
                    }
                    phrase_templates.update(chunk_templates)
                    phrase_unit_breakdowns.update(chunk_breakdowns)
                    print(
                        f"[g2pw_bundle] reuse chunk {chunk_index + 1}/{total_chunks} "
                        f"words {word_start}-{word_end - 1} templates={len(chunk_templates)}"
                    )
                    continue

        chunk_entries: Dict[str, Dict[str, Any]] = {}
        for word in chunk_words:
            entry = _make_authoritative_phrase_entry(word)
            if entry is not None:
                chunk_entries[word] = entry
        chunk_templates = {
            word: entry["templates"]
            for word, entry in chunk_entries.items()
        }
        chunk_breakdowns = {
            word: entry["unit_breakdown"]
            for word, entry in chunk_entries.items()
        }
        phrase_templates.update(chunk_templates)
        phrase_unit_breakdowns.update(chunk_breakdowns)

        if chunk_path:
            _write_progress_chunk(
                path=chunk_path,
                chunk_index=chunk_index,
                word_start=word_start,
                word_end=word_end,
                entries=chunk_entries,
                signature=signature,
            )
        print(
            f"[g2pw_bundle] built chunk {chunk_index + 1}/{total_chunks} "
            f"words {word_start}-{word_end - 1} templates={len(chunk_templates)}"
        )
    return phrase_templates, phrase_unit_breakdowns


def _build_frontend_lexicon(
    converter,
    pinyin_to_phone_map: Dict[str, List[str]],
    authoritative_progress_dir: Optional[str] = None,
    authoritative_chunk_size: int = 2000,
    authoritative_word_limit: Optional[int] = None,
) -> Dict:
    _ = list(jieba.cut("初始化"))
    tone_modifier = ToneSandhi()
    word_frequency = {
        word: int(freq)
        for word, freq in jieba.dt.FREQ.items()
        if freq and _contains_cjk(word)
    }
    word_pos = {
        word: psg.dt.word_tag_tab[word]
        for word in word_frequency
        if word in psg.dt.word_tag_tab
    }
    forced_words = set(tone_modifier.must_neural_tone_words).union(
        tone_modifier.must_not_neural_tone_words,
        must_erhua,
        not_erhua,
    )
    authoritative_phrase_words = _collect_authoritative_phrase_words(
        converter,
        word_frequency,
        forced_words,
        limit=authoritative_word_limit,
    )
    phrase_templates, phrase_unit_breakdowns = _build_phrase_phone_templates(
        converter,
        authoritative_words=authoritative_phrase_words,
        progress_dir=authoritative_progress_dir,
        chunk_size=authoritative_chunk_size,
    )
    for word in forced_words:
        if len(word) > 1 and _contains_cjk(word) and word not in word_pos:
            seg = tone_modifier.pre_merge_for_modify(psg.lcut(word))
            if len(seg) == 1 and seg[0][0] == word:
                word_pos[word] = seg[0][1]
            else:
                plain_seg = psg.lcut(word)
                if plain_seg:
                    word_pos[word] = plain_seg[-1].flag
    punctuation = sorted(set(tone_modifier.punc))
    max_word_length = max((len(word) for word in word_frequency), default=1)
    return {
        "format_version": 1,
        "segmenter": {
            "type": "jieba_frequency_dp",
            "word_frequency": word_frequency,
            "word_pos": word_pos,
            "forced_words": sorted(
                word
                for word in forced_words
                if len(word) > 1 and _contains_cjk(word)
            ),
            "posseg_hmm": _build_posseg_hmm_assets(),
            "total_frequency": int(jieba.dt.total),
            "max_word_length": int(max_word_length),
        },
        "phrase_phone_templates": phrase_templates,
        "phrase_unit_breakdowns": phrase_unit_breakdowns,
        "tone_sandhi": {
            "must_neutral_tone_words": sorted(tone_modifier.must_neural_tone_words),
            "must_not_neutral_tone_words": sorted(tone_modifier.must_not_neural_tone_words),
            "punctuation": punctuation,
        },
        "erhua": {
            "must_erhua_words": sorted(must_erhua),
            "not_erhua_words": sorted(not_erhua),
        },
        "traditional_to_simplified_map": t2s_dict,
    }


def _build_runtime_assets(
    converter,
    *,
    authoritative_progress_dir: Optional[str] = None,
    authoritative_chunk_size: int = 2000,
    authoritative_word_limit: Optional[int] = None,
) -> Dict:
    pinyin_to_phone_map = _load_pinyin_to_phone_map()
    chars = list(converter.chars)
    labels = list(converter.labels)
    phoneme_indices_by_char_id = [list(converter.char2phonemes[char]) for char in chars]
    label_phone_templates = []
    unresolved_labels = []
    for label in labels:
        template = _resolve_label_phone_template(label, converter, pinyin_to_phone_map)
        label_phone_templates.append(template)
        if template is None:
            unresolved_labels.append(label)

    monophonic_bopomofo = dict(converter.monophonic_chars_dict)
    monophonic_phone_templates = {}
    unresolved_monophonic_bopomofo = {}
    for char, label in monophonic_bopomofo.items():
        template = _resolve_label_phone_template(label, converter, pinyin_to_phone_map)
        if template is None:
            unresolved_monophonic_bopomofo[char] = label
        else:
            monophonic_phone_templates[char] = template

    char_label_aliases = {}
    label_template_by_label = {
        label: template
        for label, template in zip(labels, label_phone_templates)
        if template is not None
    }
    for char in chars:
        unresolved_labels_for_char = [
            labels[label_index]
            for label_index in converter.char2phonemes[char]
            if label_phone_templates[label_index] is None
        ]
        if not unresolved_labels_for_char:
            continue

        canonical_label = None
        for candidate_label in converter.char_bopomofo_dict.get(char, []):
            if candidate_label in label_template_by_label:
                canonical_label = candidate_label
                break
        if canonical_label is None:
            continue

        aliases = {
            unresolved_label: canonical_label
            for unresolved_label in unresolved_labels_for_char
            if unresolved_label != canonical_label
        }
        if aliases:
            char_label_aliases[char] = aliases

    return {
        "format_version": 3,
        "style": "bopomofo_phone_template",
        "chars": chars,
        "labels": labels,
        "label_phone_templates": label_phone_templates,
        "phoneme_indices_by_char_id": phoneme_indices_by_char_id,
        "polyphonic_chars": sorted(converter.polyphonic_chars_new),
        "monophonic_bopomofo": monophonic_bopomofo,
        "monophonic_phone_templates": monophonic_phone_templates,
        "char_label_aliases": char_label_aliases,
        "unresolved_labels": unresolved_labels,
        "unresolved_monophonic_bopomofo": unresolved_monophonic_bopomofo,
        "polyphonic_context_chars": int(converter.polyphonic_context_chars),
        "frontend_lexicon": _build_frontend_lexicon(
            converter,
            pinyin_to_phone_map,
            authoritative_progress_dir=authoritative_progress_dir,
            authoritative_chunk_size=authoritative_chunk_size,
            authoritative_word_limit=authoritative_word_limit,
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a Core ML bundle and runtime assets for g2pw polyphonic disambiguation."
    )
    parser.add_argument("--bundle-dir", required=True, help="Output directory for mlpackage files and manifest.json.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--coreml-compute-units",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        default="all",
    )
    parser.add_argument(
        "--coreml-minimum-deployment-target",
        choices=["macos13", "macos14", "macos15", "ios17", "ios18"],
        default="macos15",
    )
    parser.add_argument("--coreml-compute-precision", choices=["float32", "float16"], default="float32")
    parser.add_argument("--g2pw-model-dir", default="GPT_SoVITS/text/G2PWModel")
    parser.add_argument(
        "--g2pw-model-source",
        default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    )
    parser.add_argument("--g2pw-example-text", default="重庆火锅很好吃")
    parser.add_argument("--g2pw-batch-size", type=int, default=8)
    parser.add_argument("--g2pw-token-len", type=int, default=512)
    parser.add_argument(
        "--shape-mode",
        choices=["dynamic", "fixed"],
        default="dynamic",
        help="Export bounded dynamic RangeDim inputs or fixed-capacity inputs.",
    )
    parser.add_argument(
        "--runtime-assets-only",
        action="store_true",
        help="Rebuild runtime assets and manifest without re-exporting the Core ML model package.",
    )
    parser.add_argument(
        "--authoritative-chunk-size",
        type=int,
        default=2000,
        help="Chunk size used when generating authoritative Chinese phrase templates.",
    )
    parser.add_argument(
        "--authoritative-progress-dir",
        help=(
            "Optional directory for resumable authoritative phrase template chunks. "
            "Defaults to sibling build cache <bundle-dir>.build/phrase_template_progress "
            "so export-only cache files stay outside the runtime bundle."
        ),
    )
    parser.add_argument(
        "--authoritative-word-limit",
        type=int,
        help="Optional debug limit for authoritative phrase word generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.bundle_dir, exist_ok=True)
    assets_dir = os.path.join(args.bundle_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    model_filename = "g2pw.mlpackage"
    model_output_path = os.path.join(args.bundle_dir, model_filename)
    spec = get_target_spec("g2pw")
    existing_manifest = None
    manifest_path = os.path.join(args.bundle_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
    shape_mode = args.shape_mode
    if args.runtime_assets_only and existing_manifest is not None:
        shape_mode = existing_manifest.get("runtime", {}).get("coreml", {}).get("shape_mode", shape_mode)

    if args.runtime_assets_only:
        if not os.path.exists(model_output_path):
            raise FileNotFoundError(
                f"--runtime-assets-only requires an existing model package at {model_output_path}"
            )
        example_inputs = None
    else:
        wrapper, example_inputs = _build_g2pw_target(args)
        _export_coreml(
            model_output_path,
            spec,
            wrapper,
            example_inputs,
            input_types_override=_resolve_input_types_override(
                shape_mode,
                _build_g2pw_input_types,
                int(example_inputs[0].shape[0]),
                int(example_inputs[0].shape[1]),
                int(example_inputs[3].shape[1]),
            ),
            compute_units=args.coreml_compute_units,
            minimum_deployment_target=args.coreml_minimum_deployment_target,
            compute_precision=args.coreml_compute_precision,
        )

    converter = make_g2pw_converter(
        model_dir=args.g2pw_model_dir,
        model_source=args.g2pw_model_source,
        enable_non_traditional_chinese=True,
    )
    authoritative_progress_dir = _resolve_authoritative_progress_dir(
        bundle_dir=args.bundle_dir,
        assets_dir=assets_dir,
        explicit_progress_dir=args.authoritative_progress_dir,
    )
    runtime_assets = _build_runtime_assets(
        converter,
        authoritative_progress_dir=authoritative_progress_dir,
        authoritative_chunk_size=args.authoritative_chunk_size,
        authoritative_word_limit=args.authoritative_word_limit,
    )
    _cleanup_bundle_local_progress_dir(
        assets_dir=assets_dir,
        active_progress_dir=authoritative_progress_dir,
    )

    runtime_assets_filename = "g2pw_assets.json"
    runtime_assets_path = os.path.join(assets_dir, runtime_assets_filename)
    _write_json(runtime_assets_path, runtime_assets)

    tokenizer_src = os.path.join(args.g2pw_model_source, "tokenizer.json")
    if not os.path.exists(tokenizer_src):
        raise FileNotFoundError(f"Cannot locate tokenizer.json at {tokenizer_src}")
    tokenizer_filename = "tokenizer.json"
    tokenizer_output_path = os.path.join(assets_dir, tokenizer_filename)
    shutil.copyfile(tokenizer_src, tokenizer_output_path)

    if example_inputs is not None:
        batch_size = int(example_inputs[0].shape[0])
        token_len = int(example_inputs[0].shape[1])
        label_count = int(example_inputs[3].shape[1])
    elif existing_manifest is not None:
        batch_size = int(existing_manifest["runtime"]["shapes"]["batch_size"])
        token_len = int(existing_manifest["runtime"]["shapes"]["token_len"])
        label_count = int(existing_manifest["runtime"]["shapes"]["label_count"])
        batch_size_range = existing_manifest["runtime"]["shapes"].get("batch_size_range")
        token_len_range = existing_manifest["runtime"]["shapes"].get("token_len_range")
    else:
        batch_size = int(args.g2pw_batch_size)
        token_len = int(args.g2pw_token_len)
        label_count = int(len(converter.labels))
        batch_size_range = None
        token_len_range = None
    if example_inputs is not None and shape_mode == "dynamic":
        batch_size_range = _shape_range(1, batch_size)
        token_len_range = _shape_range(1, token_len)
    if shape_mode == "dynamic":
        if batch_size_range is None:
            batch_size_range = _shape_range(1, batch_size)
        if token_len_range is None:
            token_len_range = _shape_range(1, token_len)
    else:
        batch_size_range = None
        token_len_range = None
    manifest = {
        "schema_version": 1,
        "bundle_type": "gpt_sovits_g2pw_coreml_bundle",
        "bundle_dir": os.path.abspath(args.bundle_dir),
        "artifacts": {
            "model": {
                "target": "g2pw",
                "filename": model_filename,
                "path": os.path.abspath(model_output_path),
                "schema": _spec_to_schema(spec),
            },
            "tokenizer": {
                "target": "bert_wordpiece_tokenizer",
                "filename": os.path.join("assets", tokenizer_filename),
                "path": os.path.abspath(tokenizer_output_path),
            },
            "runtime_assets": {
                "target": "g2pw_runtime_assets",
                "filename": os.path.join("assets", runtime_assets_filename),
                "path": os.path.abspath(runtime_assets_path),
            },
        },
        "runtime": {
            "shapes": {
                "batch_size": batch_size,
                "batch_size_range": batch_size_range,
                "token_len": token_len,
                "token_len_range": token_len_range,
                "label_count": label_count,
            },
            "coreml": {
                "compute_units": args.coreml_compute_units,
                "minimum_deployment_target": args.coreml_minimum_deployment_target,
                "compute_precision": args.coreml_compute_precision,
                "shape_mode": shape_mode,
            },
            "driver_contract": _build_runtime_contract(
                runtime_assets["polyphonic_context_chars"],
                batch_size=batch_size,
                token_len=token_len,
                shape_mode=shape_mode,
            ),
        },
    }

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
