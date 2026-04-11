#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Example:
#   bash main/bash/eval_policy.sh --env braidedhub --policy streaming_act --policy-path <ckpt_dir>
#   bash main/bash/eval_policy.sh --env braidedhub --policy diffusion --policy-path <ckpt_dir>
#   bash main/bash/eval_policy.sh --policy act --policy-path <ckpt_dir> --dataset zeno-ai/day3_5_Exp1
#   bash main/bash/eval_policy.sh --policy diffusion --policy-path <ckpt_dir> --dataset zeno-ai/day3_5_Exp1_processed
#   bash main/bash/eval_policy.sh --env metaworld --policy act
#   bash main/bash/eval_policy.sh --env metaworld --policy streaming_act
#   bash main/bash/eval_policy.sh --env robocasa --policy act --dataset robocasa/composite/ArrangeBreadBasket --task ArrangeBreadBasket
#   bash main/bash/eval_policy.sh --env robocasa --policy act --policy-path <ckpt_dir> --tasks ArrangeBreadBasket,PickPlaceCounterToSink

python3 scripts/eval_policy.py "$@"
