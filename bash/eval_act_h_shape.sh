#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-outputs/train/h_shape_act}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/h_shape_act/${RUN_TAG}}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-20}"
MAX_STEPS="${MAX_STEPS:-120}"
FPS="${FPS:-20}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
MAX_ACTION_STEP="${MAX_ACTION_STEP:-0.3}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-0.2}"

POLICY_PATH="${POLICY_PATH:-}"
LATEST_RUN_DIR="${LATEST_RUN_DIR:-}"

if [[ $# -gt 0 && ( "${1}" == "-h" || "${1}" == "--help" ) ]]; then
  "${PYTHON_BIN}" scripts/eval_policy.py --env h_shape --policy act "$@"
  exit 0
fi

if [[ $# -gt 0 && "${1}" != -* && -z "${POLICY_PATH}" ]]; then
  POLICY_PATH="$1"
  shift
fi

POLICY_PATH="$(resolve_eval_policy_path "${POLICY_PATH}" "${LATEST_RUN_DIR}" "${TRAIN_OUTPUT_ROOT}")"
echo "Using policy path: ${POLICY_PATH}"

"${PYTHON_BIN}" scripts/eval_policy.py \
  --env h_shape \
  --policy act \
  --policy-path "${POLICY_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-rollouts "${NUM_ROLLOUTS}" \
  --max-steps "${MAX_STEPS}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  --success-threshold "${SUCCESS_THRESHOLD}" \
  --device "${DEVICE}" \
  --max-action-step "${MAX_ACTION_STEP}" \
  "$@"
