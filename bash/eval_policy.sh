#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Example:
#   bash main/bash/eval_policy.sh --env h_shape --policy act
#   bash main/bash/eval_policy.sh --env braidedhub --policy streaming_act --policy-path <ckpt_dir>

python3 scripts/eval_policy.py "$@"
