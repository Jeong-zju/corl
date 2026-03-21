#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-outputs/train/braidedhub_fourstart_streaming_act}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/braidedhub_fourstart_streaming_act/${RUN_TAG}}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"
MAX_STEPS="${MAX_STEPS:-240}"
FPS="${FPS:-20}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
MAX_ACTION_STEP="${MAX_ACTION_STEP:-2.5}"
N_ACTION_STEPS="${N_ACTION_STEPS:-5}"
SIGNATURE_BACKEND="${SIGNATURE_BACKEND:-auto}"
COLLISION_MODE="${COLLISION_MODE:-detect}"
ENABLE_RANDOMIZE="${ENABLE_RANDOMIZE:-1}"

POLICY_PATH="${POLICY_PATH:-}"
LATEST_RUN_DIR="${LATEST_RUN_DIR:-}"

if [[ $# -gt 0 && ( "${1}" == "-h" || "${1}" == "--help" ) ]]; then
  "${PYTHON_BIN}" scripts/eval_policy.py --env braidedhub --policy streaming_act "$@"
  exit 0
fi

if [[ $# -gt 0 && "${1}" != -* && -z "${POLICY_PATH}" ]]; then
  POLICY_PATH="$1"
  shift
fi

POLICY_PATH="$(resolve_eval_policy_path "${POLICY_PATH}" "${LATEST_RUN_DIR}" "${TRAIN_OUTPUT_ROOT}")"
echo "Using policy path: ${POLICY_PATH}"

RANDOMIZE_ARG=()
if [[ "${ENABLE_RANDOMIZE}" == "1" || "${ENABLE_RANDOMIZE}" == "true" || "${ENABLE_RANDOMIZE}" == "TRUE" ]]; then
  RANDOMIZE_ARG+=(--enable-randomize)
fi

"${PYTHON_BIN}" scripts/eval_policy.py \
  --env braidedhub \
  --policy streaming_act \
  --policy-path "${POLICY_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-rollouts "${NUM_ROLLOUTS}" \
  --max-steps "${MAX_STEPS}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --max-action-step "${MAX_ACTION_STEP}" \
  --collision-mode "${COLLISION_MODE}" \
  --n-action-steps "${N_ACTION_STEPS}" \
  --signature-backend "${SIGNATURE_BACKEND}" \
  "${RANDOMIZE_ARG[@]}" \
  "$@"
