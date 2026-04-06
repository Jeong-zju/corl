#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Example:
#   bash main/bash/eval_policy.sh --env braidedhub --policy streaming_act --policy-path <ckpt_dir>
#   bash main/bash/eval_policy.sh --policy act --policy-path <ckpt_dir> --dataset zeno-ai/day3_5_Exp1
#   bash main/bash/eval_policy.sh --env metaworld --policy act
#   bash main/bash/eval_policy.sh --env metaworld --policy streaming_act

python3 scripts/eval_policy.py "$@"
