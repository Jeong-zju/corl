#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Example:
#   bash main/bash/train_policy.sh --dataset zeno-ai/day3_5_Exp1 --policy act
#   bash main/bash/train_policy.sh --dataset zeno-ai/day3_5_Exp1_processed --policy streaming_act --steps 20000

python3 scripts/train_policy.py "$@"
