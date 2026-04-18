#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Example:
#   ./bash/eval_policy.sh --env braidedhub --policy streaming_act --policy-path <ckpt_dir>
#   ./bash/eval_policy.sh --env braidedhub --policy diffusion --policy-path <ckpt_dir>
#   ./bash/eval_policy.sh --policy act --policy-path <ckpt_dir> --dataset zeno-ai/day3_5_Exp1
#   ./bash/eval_policy.sh --policy diffusion --policy-path <ckpt_dir> --dataset zeno-ai/day3_5_Exp1_processed
#   ./bash/eval_policy.sh --env metaworld --policy act
#   ./bash/eval_policy.sh --env metaworld --policy streaming_act
#   ./bash/eval_policy.sh --env robocasa --policy act --dataset robocasa/composite/ArrangeBreadBasket
#   ./bash/eval_policy.sh --env robocasa --policy diffusion --dataset robocasa/atomic/CloseFridge --policy-path <ckpt_dir>
#   ./bash/eval_policy.sh --env robocasa --policy streaming_act --policy-path <ckpt_dir> --dataset robocasa/composite
#   ./bash/eval_policy.sh --env robocasa --policy act --policy-path <ckpt_dir> --tasks ArrangeBreadBasket,PickPlaceCounterToSink
#   ./bash/eval_policy.sh --env robocasa --policy act --dataset robocasa/composite/ArrangeBreadBasket --robocasa-split all

python3 scripts/eval_policy.py "$@"
