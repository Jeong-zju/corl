#!/usr/bin/env bash
set -euo pipefail

find_latest_run_dir() {
  local train_root="$1"
  local latest=""

  if [[ -d "${train_root}" ]]; then
    latest="$(
      find "${train_root}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )"
  fi

  printf '%s\n' "${latest}"
}

resolve_policy_dir_from_run() {
  local run_dir="$1"

  if [[ -f "${run_dir}/checkpoints/last/pretrained_model/model.safetensors" ]]; then
    printf '%s\n' "${run_dir}/checkpoints/last/pretrained_model"
    return 0
  fi

  if [[ -f "${run_dir}/pretrained_model/model.safetensors" ]]; then
    printf '%s\n' "${run_dir}/pretrained_model"
    return 0
  fi

  if [[ -f "${run_dir}/model.safetensors" ]]; then
    printf '%s\n' "${run_dir}"
    return 0
  fi

  return 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-outputs/train/braidedhub_fourstart_act}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/braidedhub_fourstart_act/${RUN_TAG}}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"
MAX_STEPS="${MAX_STEPS:-240}"
FPS="${FPS:-20}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
MAX_ACTION_STEP="${MAX_ACTION_STEP:-2.5}"

POLICY_PATH="${POLICY_PATH:-}"
LATEST_RUN_DIR="${LATEST_RUN_DIR:-}"

if [[ -z "${POLICY_PATH}" ]]; then
  if [[ -z "${LATEST_RUN_DIR}" ]]; then
    LATEST_RUN_DIR="$(find_latest_run_dir "${TRAIN_OUTPUT_ROOT}")"
  fi
fi

if [[ -z "${POLICY_PATH}" && -n "${LATEST_RUN_DIR}" ]]; then
  if ! POLICY_PATH="$(resolve_policy_dir_from_run "${LATEST_RUN_DIR}")"; then
    echo "Latest run does not contain policy weights: ${LATEST_RUN_DIR}" >&2
    echo "Expected one of:" >&2
    echo "  ${LATEST_RUN_DIR}/checkpoints/last/pretrained_model/model.safetensors" >&2
    echo "  ${LATEST_RUN_DIR}/pretrained_model/model.safetensors" >&2
    echo "  ${LATEST_RUN_DIR}/model.safetensors" >&2
    exit 1
  fi
fi

if [[ -z "${POLICY_PATH}" ]]; then
  echo "Could not infer latest run from TRAIN_OUTPUT_ROOT=${TRAIN_OUTPUT_ROOT}" >&2
  echo "Set LATEST_RUN_DIR or POLICY_PATH explicitly, or train a model first." >&2
  exit 1
fi

echo "Using policy path: ${POLICY_PATH}"

python3 main/scripts/eval_act_braidedhub.py \
  --policy-path "${POLICY_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-rollouts "${NUM_ROLLOUTS}" \
  --max-steps "${MAX_STEPS}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --max-action-step "${MAX_ACTION_STEP}" \
  "$@"
