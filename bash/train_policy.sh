#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Example:
#   bash main/bash/train_policy.sh --env braidedhub --policy streaming_act
#   bash main/bash/train_policy.sh --env braidedhub --policy diffusion
#   bash main/bash/train_policy.sh --dataset zeno-ai/day3_5_Exp1 --policy act
#   bash main/bash/train_policy.sh --dataset zeno-ai/day3_5_Exp1_processed --policy diffusion
#   bash main/bash/train_policy.sh --dataset zeno-ai/day3_5_Exp1_processed --policy streaming_act --steps 20000
#   bash main/bash/train_policy.sh --dataset metaworld_mt50 --policy act
#   bash main/bash/train_policy.sh --dataset metaworld_mt50 --policy streaming_act

policy_name="act"
env_name=""
dataset_name=""
forward_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --policy requires a value." >&2
        exit 2
      fi
      policy_name="$2"
      forward_args+=("$1" "$2")
      shift 2
      ;;
    --policy=*)
      policy_name="${1#*=}"
      forward_args+=("$1")
      shift
      ;;
    --env)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --env requires a value." >&2
        exit 2
      fi
      env_name="$2"
      shift 2
      ;;
    --env=*)
      env_name="${1#*=}"
      shift
      ;;
    --dataset)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --dataset requires a value." >&2
        exit 2
      fi
      dataset_name="$2"
      forward_args+=("$1" "$2")
      shift 2
      ;;
    --dataset=*)
      dataset_name="${1#*=}"
      forward_args+=("$1")
      shift
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${dataset_name}" && -n "${env_name}" ]]; then
  defaults_path="bash/defaults/${env_name}/${policy_name}.yaml"
  if [[ ! -f "${defaults_path}" ]]; then
    echo "[ERROR] Could not resolve dataset from env defaults." >&2
    echo "  env=${env_name}" >&2
    echo "  policy=${policy_name}" >&2
    echo "  expected defaults file: ${defaults_path}" >&2
    echo "Pass --dataset explicitly or add the defaults file." >&2
    exit 2
  fi

  dataset_name="$(
    python3 -c '
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
train_cfg = data.get("train", {})
dataset_root = train_cfg.get("dataset_root")
if not dataset_root:
    raise SystemExit(1)
print(str(dataset_root))
' "${defaults_path}"
  )"

  if [[ -z "${dataset_name}" ]]; then
    echo "[ERROR] Failed to read train.dataset_root from ${defaults_path}." >&2
    exit 2
  fi

  forward_args+=("--dataset" "${dataset_name}")
fi

python3 scripts/train_policy.py "${forward_args[@]}"
