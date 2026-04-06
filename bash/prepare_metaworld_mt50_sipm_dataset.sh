#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Example:
#   bash main/bash/prepare_metaworld_mt50_sipm_dataset.sh
#   bash main/bash/prepare_metaworld_mt50_sipm_dataset.sh --workers 4
#   bash main/bash/prepare_metaworld_mt50_sipm_dataset.sh --signature-backend simple
#   bash main/bash/prepare_metaworld_mt50_sipm_dataset.sh --source-dataset lerobot_metaworld_mt50_video --target-dataset data/lerobot_metaworld_mt50_video_sipm

source_dataset="lerobot_metaworld_mt50"
target_dataset="data/lerobot_metaworld_mt50_sipm"
workers=""
signature_backend="auto"
signature_depth="3"
overwrite_output=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dataset)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --source-dataset requires a value." >&2
        exit 2
      fi
      source_dataset="$2"
      shift 2
      ;;
    --source-dataset=*)
      source_dataset="${1#*=}"
      shift
      ;;
    --target-dataset)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --target-dataset requires a value." >&2
        exit 2
      fi
      target_dataset="$2"
      shift 2
      ;;
    --target-dataset=*)
      target_dataset="${1#*=}"
      shift
      ;;
    --workers)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --workers requires a value." >&2
        exit 2
      fi
      workers="$2"
      shift 2
      ;;
    --workers=*)
      workers="${1#*=}"
      shift
      ;;
    --signature-backend)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --signature-backend requires a value." >&2
        exit 2
      fi
      signature_backend="$2"
      shift 2
      ;;
    --signature-backend=*)
      signature_backend="${1#*=}"
      shift
      ;;
    --signature-depth)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --signature-depth requires a value." >&2
        exit 2
      fi
      signature_depth="$2"
      shift 2
      ;;
    --signature-depth=*)
      signature_depth="${1#*=}"
      shift
      ;;
    --overwrite-output)
      overwrite_output=true
      shift
      ;;
    *)
      echo "[ERROR] Unsupported argument: $1" >&2
      exit 2
      ;;
  esac
done

cmd=(
  python3
  data/process_dataset.py
  "${source_dataset}"
  --operations
  update-signatures
  --signature-type
  both
  --path-signature-window-size
  0
  --path-signature-depth
  "${signature_depth}"
  --signature-backend
  "${signature_backend}"
  --output-dir
  "${target_dataset}"
)

if [[ -n "${workers}" ]]; then
  cmd+=(--workers "${workers}")
fi

if [[ "${overwrite_output}" == true ]]; then
  cmd+=(--overwrite-output)
fi

printf 'Preparing Meta-World MT50 SIPM dataset\n'
printf '  source: %s\n' "${source_dataset}"
printf '  target: %s\n' "${target_dataset}"
printf '  signature_depth: %s\n' "${signature_depth}"
printf '  signature_backend: %s\n' "${signature_backend}"
if [[ -n "${workers}" ]]; then
  printf '  workers: %s\n' "${workers}"
fi

"${cmd[@]}"
