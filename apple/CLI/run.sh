#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLI_BIN="${GPTSOVITS_CLI_BIN:-/Volumes/1T/cache/swift-bin/gpt-sovits-cli}"
TMPDIR_DEFAULT="/Volumes/1T/cache/tmp"
OUTPUT_ROOT_DEFAULT="/Volumes/1T/cache/GPTSoVITS-artifacts/cli"
COMPUTE_UNITS_DEFAULT="${GPTSOVITS_CLI_COMPUTE_UNITS:-cpu_and_ne}"

print_wrapper_help() {
    cat <<EOF
Usage:
  bash apple/CLI/run.sh [gpt-sovits-cli options]
  bash apple/CLI/run.sh --build-only
  bash apple/CLI/run.sh --help-wrapper

Wrapper behavior:
  - Automatically builds / refreshes ${CLI_BIN} when sources are newer.
  - Defaults TMPDIR to ${TMPDIR_DEFAULT}.
  - Defaults --compute-units to ${COMPUTE_UNITS_DEFAULT} when omitted.
  - Auto-fills --output-wav and --summary-json under ${OUTPUT_ROOT_DEFAULT} when omitted.
  - Auto-resolves --bundle-dir from GPTSOVITS_BUNDLE_DIR or local artifacts/coreml/chinese_synthesis_bundle* when omitted.

Forwarded help:
  bash apple/CLI/run.sh --help

Examples:
  bash apple/CLI/run.sh \\
    --bundle-dir /absolute/path/to/chinese_synthesis_bundle \\
    --prompt-text-file test.lab \\
    --target-text "你好，这是仓库内的直接 CLI 入口。" \\
    --reference-audio test.wav

  bash apple/CLI/run.sh \\
    --bundle-dir /absolute/path/to/chinese_synthesis_bundle \\
    --prompt-text-file test.lab \\
    --target-text "Swift CLI now supports cross language output." \\
    --prompt-language zh \\
    --target-language en \\
    --reference-audio test.wav
EOF
}

ensure_value() {
    local option="$1"
    local value="${2:-}"
    if [[ -z "$value" ]]; then
        echo "Missing value for ${option}" >&2
        exit 1
    fi
}

detect_needs_build() {
    if [[ ! -x "$CLI_BIN" ]]; then
        return 0
    fi
    if [[ "${SCRIPT_DIR}/build.sh" -nt "$CLI_BIN" || "${SCRIPT_DIR}/GPTSoVITSCLI.swift" -nt "$CLI_BIN" || "$0" -nt "$CLI_BIN" ]]; then
        return 0
    fi

    local newer_source
    newer_source="$(find "${REPO_ROOT}/apple/Sources" -name '*.swift' -newer "$CLI_BIN" -print -quit)"
    [[ -n "$newer_source" ]]
}

build_cli() {
    mkdir -p "$(dirname "$CLI_BIN")"
    TMPDIR="${TMPDIR:-$TMPDIR_DEFAULT}" bash "${SCRIPT_DIR}/build.sh" "$CLI_BIN" >/dev/null
}

has_forwarded_flag() {
    local needle="$1"
    local argument
    for argument in "${FORWARDED_ARGS[@]}"; do
        if [[ "$argument" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

resolve_bundle_dir() {
    if [[ -n "${EXPLICIT_BUNDLE_DIR:-}" ]]; then
        printf '%s\n' "$EXPLICIT_BUNDLE_DIR"
        return 0
    fi
    if [[ -n "${GPTSOVITS_BUNDLE_DIR:-}" && -d "${GPTSOVITS_BUNDLE_DIR}" ]]; then
        printf '%s\n' "${GPTSOVITS_BUNDLE_DIR}"
        return 0
    fi

    local preferred_candidates=(
        "${REPO_ROOT}/artifacts/coreml/chinese_synthesis_bundle"
        "${REPO_ROOT}/artifacts/coreml/chinese_synthesis_bundle_fp32_stable_multilingual"
        "${REPO_ROOT}/artifacts/coreml/chinese_synthesis_bundle_prompt264_stable"
    )
    local candidate
    for candidate in "${preferred_candidates[@]}"; do
        if [[ -d "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    candidate="$(find "${REPO_ROOT}/artifacts/coreml" -maxdepth 1 -type d -name 'chinese_synthesis_bundle*' | sort | head -n 1)"
    if [[ -n "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
    fi

    return 1
}

WRAPPER_HELP=0
BUILD_ONLY=0
EXPLICIT_BUNDLE_DIR=""
EXPLICIT_OUTPUT_WAV=""
EXPLICIT_SUMMARY_JSON=""
EXPLICIT_COMPUTE_UNITS=""
FORWARDED_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
    --help-wrapper)
        WRAPPER_HELP=1
        shift
        ;;
    --build-only)
        BUILD_ONLY=1
        shift
        ;;
    --bundle-dir)
        ensure_value "$1" "${2:-}"
        EXPLICIT_BUNDLE_DIR="$2"
        FORWARDED_ARGS+=("$1" "$2")
        shift 2
        ;;
    --output-wav)
        ensure_value "$1" "${2:-}"
        EXPLICIT_OUTPUT_WAV="$2"
        FORWARDED_ARGS+=("$1" "$2")
        shift 2
        ;;
    --summary-json)
        ensure_value "$1" "${2:-}"
        EXPLICIT_SUMMARY_JSON="$2"
        FORWARDED_ARGS+=("$1" "$2")
        shift 2
        ;;
    --compute-units)
        ensure_value "$1" "${2:-}"
        EXPLICIT_COMPUTE_UNITS="$2"
        FORWARDED_ARGS+=("$1" "$2")
        shift 2
        ;;
    *)
        FORWARDED_ARGS+=("$1")
        shift
        ;;
    esac
done

if [[ "$WRAPPER_HELP" -eq 1 ]]; then
    print_wrapper_help
    exit 0
fi

export TMPDIR="${TMPDIR:-$TMPDIR_DEFAULT}"
mkdir -p "$TMPDIR" "$OUTPUT_ROOT_DEFAULT"

if detect_needs_build; then
    build_cli
fi

if [[ "$BUILD_ONLY" -eq 1 ]]; then
    printf '%s\n' "$CLI_BIN"
    exit 0
fi

if [[ ${#FORWARDED_ARGS[@]} -eq 0 ]]; then
    print_wrapper_help
    exit 0
fi

if has_forwarded_flag "--help" || has_forwarded_flag "-h"; then
    exec "$CLI_BIN" "${FORWARDED_ARGS[@]}"
fi

if ! printf '%s\n' "${FORWARDED_ARGS[@]}" | grep -qx -- '--bundle-dir'; then
    if RESOLVED_BUNDLE_DIR="$(resolve_bundle_dir)"; then
        FORWARDED_ARGS=(--bundle-dir "$RESOLVED_BUNDLE_DIR" "${FORWARDED_ARGS[@]}")
    else
        cat <<EOF >&2
Unable to resolve --bundle-dir automatically.
Pass --bundle-dir explicitly, or set GPTSOVITS_BUNDLE_DIR to a valid top-level synthesis bundle directory.
Looked under:
  ${REPO_ROOT}/artifacts/coreml/chinese_synthesis_bundle*
EOF
        exit 1
    fi
fi

RESOLVED_OUTPUT_WAV="$EXPLICIT_OUTPUT_WAV"
if [[ -z "$RESOLVED_OUTPUT_WAV" ]]; then
    TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
    UNIQUE_SUFFIX="${TIMESTAMP}_$$_${RANDOM}"
    RESOLVED_OUTPUT_WAV="${OUTPUT_ROOT_DEFAULT}/${UNIQUE_SUFFIX}.wav"
    FORWARDED_ARGS+=(--output-wav "$RESOLVED_OUTPUT_WAV")
fi

if [[ -z "$EXPLICIT_SUMMARY_JSON" ]]; then
    if [[ "$RESOLVED_OUTPUT_WAV" == *.wav ]]; then
        RESOLVED_SUMMARY_JSON="${RESOLVED_OUTPUT_WAV%.wav}.json"
    else
        RESOLVED_SUMMARY_JSON="${RESOLVED_OUTPUT_WAV}.json"
    fi
    FORWARDED_ARGS+=(--summary-json "$RESOLVED_SUMMARY_JSON")
fi

if [[ -z "$EXPLICIT_COMPUTE_UNITS" ]]; then
    FORWARDED_ARGS+=(--compute-units "$COMPUTE_UNITS_DEFAULT")
fi

exec "$CLI_BIN" "${FORWARDED_ARGS[@]}"
