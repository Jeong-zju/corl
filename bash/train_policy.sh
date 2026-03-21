#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Example:
#   bash main/bash/train_policy.sh --env braidedhub --policy act
#   bash main/bash/train_policy.sh --env braidedhub --policy streaming_act --steps 20000

python3 scripts/train_policy.py "$@"
