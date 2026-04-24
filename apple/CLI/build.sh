#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_PATH="${1:-/Volumes/1T/cache/swift-bin/gpt-sovits-cli}"
SWIFTC_BIN="${SWIFTC_BIN:-swiftc}"
TMPDIR_DEFAULT="/Volumes/1T/cache/tmp"

export TMPDIR="${TMPDIR:-$TMPDIR_DEFAULT}"
mkdir -p "$TMPDIR"
mkdir -p "$(dirname "$OUTPUT_PATH")"

SOURCE_FILES=()
while IFS= read -r file; do
    SOURCE_FILES+=("$file")
done < <(find "${REPO_ROOT}/apple/Sources" -name '*.swift' | sort)

"$SWIFTC_BIN" -O \
    -framework AVFoundation \
    -framework NaturalLanguage \
    -framework Accelerate \
    "${SOURCE_FILES[@]}" \
    "${REPO_ROOT}/apple/CLI/GPTSoVITSCLI.swift" \
    -o "$OUTPUT_PATH"

echo "$OUTPUT_PATH"
