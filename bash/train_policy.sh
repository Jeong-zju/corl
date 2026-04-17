#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Example:
#   ./bash/train_policy.sh --env braidedhub --policy streaming_act
#   ./bash/train_policy.sh --env braidedhub --policy diffusion
#   ./bash/train_policy.sh --dataset zeno-ai/day3_5_Exp1 --policy act
#   ./bash/train_policy.sh --dataset zeno-ai/day3_5_Exp1_processed --policy diffusion
#   ./bash/train_policy.sh --dataset zeno-ai/day3_5_Exp1_processed --policy streaming_act --steps 20000
#   ./bash/train_policy.sh --dataset zeno-ai/day3_5_Exp1_processed --policy streaming_act --resume --steps 40000
#   ./bash/train_policy.sh --dataset metaworld_mt50 --policy act
#   ./bash/train_policy.sh --dataset metaworld_mt50 --policy streaming_act
#   ./bash/train_policy.sh --dataset robocasa/composite/ArrangeBreadBasket --policy act
#   ./bash/train_policy.sh --env robocasa/composite --policy streaming_act
#   ./bash/train_policy.sh --env robocasa/composite/ArrangeBreadBasket --policy act

policy_name="act"
env_name=""
dataset_name=""
defaults_path=""
forward_args=()

resolved_defaults_dataset_root=""
distributed_enabled="false"
distributed_launcher="accelerate"
distributed_num_processes="1"
distributed_gpu_ids=""
distributed_num_machines="1"
distributed_machine_rank="0"
distributed_main_process_ip=""
distributed_main_process_port=""

resolve_defaults_path_from_dataset() {
  local dataset_selector="$1"
  python3 -c '
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
dataset_selector = sys.argv[2]
policy_name = sys.argv[3]

sys.path.insert(0, str(repo_root / "scripts"))
from policy_defaults import resolve_dataset_defaults_path

path = resolve_dataset_defaults_path(dataset_selector, policy_name)
print("" if path is None else path.as_posix())
' "${REPO_ROOT}" "${dataset_selector}" "${policy_name}"
}

load_train_defaults() {
  local path="$1"
  mapfile -t _train_defaults < <(
    python3 -c '
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
train_cfg = data.get("train", {})
if not isinstance(train_cfg, dict):
    train_cfg = {}
distributed_cfg = train_cfg.get("distributed", {})
if not isinstance(distributed_cfg, dict):
    distributed_cfg = {}

values = [
    str(train_cfg.get("dataset_root", "")),
    str(bool(distributed_cfg.get("enabled", False))).lower(),
    str(distributed_cfg.get("launcher", "accelerate")),
    str(distributed_cfg.get("num_processes", 1)),
    "" if distributed_cfg.get("gpu_ids") is None else str(distributed_cfg.get("gpu_ids")),
    str(distributed_cfg.get("num_machines", 1)),
    str(distributed_cfg.get("machine_rank", 0)),
    "" if distributed_cfg.get("main_process_ip") is None else str(distributed_cfg.get("main_process_ip")),
    "" if distributed_cfg.get("main_process_port") is None else str(distributed_cfg.get("main_process_port")),
]
print("\n".join(values))
' "${path}"
  )

  resolved_defaults_dataset_root="${_train_defaults[0]:-}"
  distributed_enabled="${_train_defaults[1]:-false}"
  distributed_launcher="${_train_defaults[2]:-accelerate}"
  distributed_num_processes="${_train_defaults[3]:-1}"
  distributed_gpu_ids="${_train_defaults[4]:-}"
  distributed_num_machines="${_train_defaults[5]:-1}"
  distributed_machine_rank="${_train_defaults[6]:-0}"
  distributed_main_process_ip="${_train_defaults[7]:-}"
  distributed_main_process_port="${_train_defaults[8]:-}"
}

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

if [[ -n "${dataset_name}" ]]; then
  defaults_path="$(resolve_defaults_path_from_dataset "${dataset_name}")"
elif [[ -n "${env_name}" ]]; then
  defaults_path="bash/defaults/${env_name}/${policy_name}.yaml"
  if [[ ! -f "${defaults_path}" ]]; then
    echo "[ERROR] Could not resolve dataset from env defaults." >&2
    echo "  env=${env_name}" >&2
    echo "  policy=${policy_name}" >&2
    echo "  expected defaults file: ${defaults_path}" >&2
    echo "Pass --dataset explicitly or add the defaults file." >&2
    exit 2
  fi
fi

if [[ -n "${defaults_path}" && -f "${defaults_path}" ]]; then
  load_train_defaults "${defaults_path}"
fi

if [[ -z "${dataset_name}" && -n "${env_name}" ]]; then
  dataset_name="${resolved_defaults_dataset_root}"

  if [[ -z "${dataset_name}" ]]; then
    echo "[ERROR] Failed to read train.dataset_root from ${defaults_path}." >&2
    exit 2
  fi

  forward_args+=("--dataset" "${dataset_name}")
fi

if [[ -z "${CORL_TRAIN_RUN_STAMP:-}" ]]; then
  export CORL_TRAIN_RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
fi

train_cmd=(python3 scripts/train_policy.py "${forward_args[@]}")

if [[ "${distributed_enabled}" == "true" ]]; then
  if [[ "${distributed_launcher}" != "accelerate" ]]; then
    echo "[ERROR] Unsupported train.distributed.launcher=${distributed_launcher}." >&2
    echo "Only \"accelerate\" is supported right now." >&2
    exit 2
  fi

  if ! [[ "${distributed_num_processes}" =~ ^[0-9]+$ ]] || [[ "${distributed_num_processes}" -lt 1 ]]; then
    echo "[ERROR] train.distributed.num_processes must be a positive integer." >&2
    exit 2
  fi

  if (( distributed_num_processes > 1 )); then
    python3 -c 'import accelerate' >/dev/null 2>&1 || {
      echo "[ERROR] Multi-GPU launch requires the Python package \"accelerate\"." >&2
      echo "Install dependencies from requirements.txt/environment.yml and retry." >&2
      exit 1
    }

    if [[ -n "${distributed_gpu_ids}" && "${distributed_gpu_ids}" != "all" ]]; then
      export CUDA_VISIBLE_DEVICES="${distributed_gpu_ids}"
    fi

    train_cmd=(
      python3 -m accelerate.commands.launch
      --multi_gpu
      --num_processes "${distributed_num_processes}"
      --num_machines "${distributed_num_machines}"
      --machine_rank "${distributed_machine_rank}"
    )

    if [[ -n "${distributed_main_process_ip}" ]]; then
      train_cmd+=(--main_process_ip "${distributed_main_process_ip}")
    fi
    if [[ -n "${distributed_main_process_port}" ]]; then
      train_cmd+=(--main_process_port "${distributed_main_process_port}")
    fi

    train_cmd+=(scripts/train_policy.py "${forward_args[@]}")
  fi
fi

echo "[INFO] train_policy launcher:"
echo "  defaults_path=${defaults_path:-<none>}"
echo "  distributed_enabled=${distributed_enabled}"
echo "  num_processes=${distributed_num_processes}"
if [[ -n "${distributed_gpu_ids}" ]]; then
  echo "  gpu_ids=${distributed_gpu_ids}"
fi

"${train_cmd[@]}"
