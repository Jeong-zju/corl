#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-data/zeno-ai/braidedhub_fourstart_implicit_cue_v30}"
DATASET_REPO_ID="${DATASET_REPO_ID:-zeno-ai/braidedhub_fourstart_implicit_cue_v30}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/train/braidedhub_fourstart_act}"
JOB_NAME="${JOB_NAME:-act_braidedhub_fourstart}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot-braidedhub-act}"
WANDB_MODE="${WANDB_MODE:-online}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
LOG_FREQ="${LOG_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
EVAL_FREQ="${EVAL_FREQ:--1}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-10}"

export WANDB_CONSOLE
export WANDB__SERVICE_WAIT

python3 main/scripts/train_act_rrt_wandb.py \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-repo-id "${DATASET_REPO_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --job-name "${JOB_NAME}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --log-freq "${LOG_FREQ}" \
  --save-freq "${SAVE_FREQ}" \
  --eval-freq "${EVAL_FREQ}" \
  --device "${DEVICE}" \
  --chunk-size "${CHUNK_SIZE}" \
  --disable-imagenet-stats \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-mode "${WANDB_MODE}" \
  "$@"
