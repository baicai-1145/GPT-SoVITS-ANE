#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR" || exit 1

RESET="\033[0m"
BOLD="\033[1m"
ERROR="\033[1;31m[ERROR]: $RESET"
WARNING="\033[1;33m[WARNING]: $RESET"
INFO="\033[1;32m[INFO]: $RESET"
SUCCESS="\033[1;34m[SUCCESS]: $RESET"

set -eEuo pipefail
set -o errtrace

trap 'on_error $LINENO "$BASH_COMMAND" $?' ERR

on_error() {
    local lineno="$1"
    local cmd="$2"
    local code="$3"

    echo -e "${ERROR}${BOLD}Command \"${cmd}\" failed${RESET} at ${BOLD}line ${lineno}${RESET} with exit code ${BOLD}${code}${RESET}"
    exit "$code"
}

SOURCE=""
MODEL_VERSION=""
ARTIFACTS_ROOT="artifacts/coreml"
MANIFEST_PATH="apple/runtime_assets_manifest.json"
COREML_REPO_ID=""
UPSTREAM_PRETRAINED_REPO_ID="XXXXRT/GPT-SoVITS-Pretrained"

print_help() {
    echo "Usage: bash download.sh [OPTIONS]"
    echo ""
    echo "Dedicated downloader for prebuilt Apple Core ML runtime assets."
    echo ""
    echo "Options:"
    echo "  --source         hf|hf-mirror|modelscope    Download source for prebuilt Core ML bundles (REQUIRED)"
    echo "  --version        v1|v2|v2Pro|v2ProPlus|all  Runtime bundle version to install (REQUIRED)"
    echo "  --artifacts-root PATH                       Local install root. Default: artifacts/coreml"
    echo "  --coreml-repo-id REPO_ID                    Override default runtime asset repo id"
    echo "  --manifest PATH                             Override runtime asset manifest path"
    echo "  -h, --help                                  Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  bash download.sh --source hf --version v2ProPlus"
    echo "  bash download.sh --source hf-mirror --version all --artifacts-root /Volumes/1T/GPTSoVITS/coreml"
    echo "  bash download.sh --source modelscope --version v2ProPlus"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --source)
        case "${2:-}" in
        hf | HF)
            SOURCE="hf"
            ;;
        hf-mirror | HF-Mirror)
            SOURCE="hf-mirror"
            ;;
        modelscope | ModelScope)
            SOURCE="modelscope"
            ;;
        *)
            echo -e "${ERROR}Invalid source: ${2:-<missing>}"
            print_help
            exit 1
            ;;
        esac
        shift 2
        ;;
    --version)
        case "${2:-}" in
        v1 | v2 | v2Pro | v2ProPlus | all)
            MODEL_VERSION="$2"
            ;;
        *)
            echo -e "${ERROR}Invalid version: ${2:-<missing>}"
            print_help
            exit 1
            ;;
        esac
        shift 2
        ;;
    --artifacts-root)
        ARTIFACTS_ROOT="${2:-}"
        shift 2
        ;;
    --coreml-repo-id)
        COREML_REPO_ID="${2:-}"
        shift 2
        ;;
    --manifest)
        MANIFEST_PATH="${2:-}"
        shift 2
        ;;
    -h | --help)
        print_help
        exit 0
        ;;
    *)
        echo -e "${ERROR}Unknown argument: $1"
        print_help
        exit 1
        ;;
    esac
done

if [[ -z "$SOURCE" || -z "$MODEL_VERSION" ]]; then
    print_help
    exit 1
fi

if [[ "$ARTIFACTS_ROOT" != /* ]]; then
    ARTIFACTS_ROOT="$SCRIPT_DIR/$ARTIFACTS_ROOT"
fi
if [[ "$MANIFEST_PATH" != /* ]]; then
    MANIFEST_PATH="$SCRIPT_DIR/$MANIFEST_PATH"
fi

if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo -e "${ERROR}Cannot find runtime asset manifest: $MANIFEST_PATH"
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    if command -v python &>/dev/null; then
        PYTHON_BIN="python"
    else
        echo -e "${ERROR}python3 or python is required to read ${MANIFEST_PATH}"
        exit 1
    fi
fi

if ! command -v curl &>/dev/null; then
    echo -e "${ERROR}curl is required"
    exit 1
fi

if [[ "$(uname)" == "Darwin" ]]; then
    if ! xcode-select -p &>/dev/null; then
        echo -e "${WARNING}Xcode Command Line Tools are not installed."
        echo -e "${WARNING}Swift/Core ML runtime compilation may fail until you run: xcode-select --install"
    fi
else
    echo -e "${WARNING}download.sh is designed for Apple Core ML runtime assets."
    echo -e "${WARNING}Current host: $(uname -s) $(uname -m)"
fi

if [[ -z "$COREML_REPO_ID" ]]; then
    COREML_REPO_ID="$("$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload["coreml_repo_id"])
PY
)"
fi

UPSTREAM_TOKENIZER_PATH="$("$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload["upstream_tokenizer_path"])
PY
)"

case "$SOURCE" in
hf)
    COREML_RESOLVE_PREFIX="https://huggingface.co/${COREML_REPO_ID}/resolve/main"
    UPSTREAM_RESOLVE_PREFIX="https://huggingface.co/${UPSTREAM_PRETRAINED_REPO_ID}/resolve/main"
    ;;
hf-mirror)
    COREML_RESOLVE_PREFIX="https://hf-mirror.com/${COREML_REPO_ID}/resolve/main"
    UPSTREAM_RESOLVE_PREFIX="https://hf-mirror.com/${UPSTREAM_PRETRAINED_REPO_ID}/resolve/main"
    ;;
modelscope)
    COREML_RESOLVE_PREFIX="https://www.modelscope.cn/models/${COREML_REPO_ID}/resolve/master"
    UPSTREAM_RESOLVE_PREFIX="https://www.modelscope.cn/models/${UPSTREAM_PRETRAINED_REPO_ID}/resolve/master"
    ;;
esac

read_manifest_list() {
    local mode="$1"
    "$PYTHON_BIN" - "$MANIFEST_PATH" "$mode" <<'PY'
import json, sys

manifest_path = sys.argv[1]
mode = sys.argv[2]
with open(manifest_path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)

if mode == "common":
    items = payload["common_repo_files"]
elif mode.startswith("version:"):
    version = mode.split(":", 1)[1]
    items = payload["version_repo_files"][version]
else:
    raise SystemExit(f"unknown manifest mode: {mode}")

for item in items:
    print(item)
PY
}

download_file() {
    local url="$1"
    local destination="$2"

    if [[ -f "$destination" ]]; then
        echo -e "${INFO}File exists: ${destination}"
        return
    fi

    mkdir -p "$(dirname "$destination")"
    echo -e "${INFO}Downloading $(basename "$destination")"
    curl --fail --location --retry 5 --retry-delay 3 --continue-at - --output "${destination}.part" "$url"
    mv "${destination}.part" "$destination"
    echo -e "${SUCCESS}Downloaded ${destination}"
}

download_common_assets() {
    local repo_path destination
    while IFS= read -r repo_path; do
        [[ -n "$repo_path" ]] || continue
        destination="$ARTIFACTS_ROOT/$repo_path"
        download_file "${COREML_RESOLVE_PREFIX}/${repo_path}" "$destination"
    done < <(read_manifest_list common)

    download_file "${UPSTREAM_RESOLVE_PREFIX}/${UPSTREAM_TOKENIZER_PATH}" "${ARTIFACTS_ROOT}/tokenizer.json"
}

download_version_assets() {
    local version="$1"
    local relative_path destination
    while IFS= read -r relative_path; do
        [[ -n "$relative_path" ]] || continue
        destination="$ARTIFACTS_ROOT/pretrained_model_bundles/${version}/${relative_path}"
        download_file "${COREML_RESOLVE_PREFIX}/pretrained_model_bundles/${version}/${relative_path}" "$destination"
    done < <(read_manifest_list "version:${version}")
}

configure_selected_version_links() {
    local version="$1"
    local t2s_link="$ARTIFACTS_ROOT/t2s_bundle_monolithic_fp32_stable"
    local vits_link="$ARTIFACTS_ROOT/vits_bundle_fp32_stable"

    if [[ -e "$t2s_link" && ! -L "$t2s_link" ]]; then
        echo -e "${WARNING}Skip updating ${t2s_link}; a real directory already exists."
    else
        ln -sfn "pretrained_model_bundles/${version}/t2s_bundle_monolithic_fp32_stable" "$t2s_link"
    fi

    if [[ -e "$vits_link" && ! -L "$vits_link" ]]; then
        echo -e "${WARNING}Skip updating ${vits_link}; a real directory already exists."
    else
        ln -sfn "pretrained_model_bundles/${version}/vits_bundle_fp32_stable" "$vits_link"
    fi
}

clear_selected_version_links() {
    rm -f "$ARTIFACTS_ROOT/t2s_bundle_monolithic_fp32_stable"
    rm -f "$ARTIFACTS_ROOT/vits_bundle_fp32_stable"
}

echo -e "${INFO}Downloading prebuilt Core ML runtime assets into ${ARTIFACTS_ROOT}"
echo -e "${INFO}Download source: ${SOURCE}"
echo -e "${INFO}Core ML bundle repo: ${COREML_REPO_ID}"

mkdir -p "$ARTIFACTS_ROOT/pretrained_model_bundles"
download_common_assets

if [[ "$MODEL_VERSION" == "all" ]]; then
    for version in v1 v2 v2Pro v2ProPlus; do
        echo -e "${INFO}Downloading version bundle: ${version}"
        download_version_assets "$version"
    done
    clear_selected_version_links
    echo -e "${INFO}Downloaded all versions; no single-version t2s/vits symlink was updated."
else
    echo -e "${INFO}Downloading version bundle: ${MODEL_VERSION}"
    download_version_assets "$MODEL_VERSION"
    configure_selected_version_links "$MODEL_VERSION"
fi

echo -e "${SUCCESS}Core ML runtime assets are ready."
echo -e "${INFO}Common assets root: ${ARTIFACTS_ROOT}"
echo -e "${INFO}Tokenizer path: ${ARTIFACTS_ROOT}/tokenizer.json"
echo -e "${INFO}Version bundles root: ${ARTIFACTS_ROOT}/pretrained_model_bundles"
if [[ "$MODEL_VERSION" != "all" ]]; then
    echo -e "${INFO}Selected T2S bundle: ${ARTIFACTS_ROOT}/t2s_bundle_monolithic_fp32_stable"
    echo -e "${INFO}Selected VITS bundle: ${ARTIFACTS_ROOT}/vits_bundle_fp32_stable"
fi
echo -e "${INFO}This downloader does not install PyTorch, requirements.txt, NLTK, OpenJTalk, or raw S1/S2 checkpoints."
