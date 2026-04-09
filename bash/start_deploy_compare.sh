#!/usr/bin/env bash
set -euo pipefail

ORIGINAL_CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_workspace_root() {
  local script_dir="$1"
  local -a candidates=(
    "$(cd "${script_dir}/.." && pwd)"
    "$(cd "${script_dir}/../.." && pwd)"
    "${PWD}"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/main/deploy/ros1_adapter/ros1_adapter_node.py" ]]; then
      echo "${candidate}"
      return 0
    fi
    if [[ -f "${candidate}/deploy/ros1_adapter/ros1_adapter_node.py" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

WORKSPACE_ROOT="$(resolve_workspace_root "${SCRIPT_DIR}")" || {
  echo "[ERROR] Failed to resolve workspace root from ${SCRIPT_DIR}." >&2
  exit 2
}
cd "${WORKSPACE_ROOT}"

if [[ -d "${WORKSPACE_ROOT}/main/deploy" ]]; then
  DEPLOY_PREFIX="main/deploy"
elif [[ -d "${WORKSPACE_ROOT}/deploy" ]]; then
  DEPLOY_PREFIX="deploy"
else
  echo "[ERROR] Could not locate deploy directory under ${WORKSPACE_ROOT}." >&2
  exit 2
fi

resolve_user_path() {
  local raw_path="$1"
  local create_ok="${2:-false}"
  python3 - "${ORIGINAL_CWD}" "${WORKSPACE_ROOT}" "${raw_path}" "${create_ok}" <<'PY'
from pathlib import Path
import sys

original_cwd = Path(sys.argv[1])
workspace_root = Path(sys.argv[2])
raw = sys.argv[3]
create_ok = sys.argv[4].strip().lower() == "true"

raw_path = Path(raw).expanduser()
if raw_path.is_absolute():
    print(str(raw_path.resolve(strict=False)))
    raise SystemExit(0)

cwd_candidate = (original_cwd / raw_path).resolve(strict=False)
workspace_candidate = (workspace_root / raw_path).resolve(strict=False)

if cwd_candidate.exists():
    print(str(cwd_candidate))
elif workspace_candidate.exists():
    print(str(workspace_candidate))
elif create_ok:
    print(str(cwd_candidate))
else:
    print(str(cwd_candidate))
PY
}

ACT_CONFIG="${DEPLOY_PREFIX}/configs/deploy_zeno_compare_act.yaml"
STREAMING_ACT_CONFIG="${DEPLOY_PREFIX}/configs/deploy_zeno_compare_streaming_act.yaml"
LOG_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash main/bash/start_deploy_compare.sh \
    [--act-config <compare_act_yaml>] \
    [--streaming-act-config <compare_streaming_act_yaml>] \
    [--log-dir /tmp/corl_deploy_compare]

This script starts 2 ROS1 deploy nodes:
  1. ACT single-process deploy node
  2. Streaming ACT single-process deploy node

Each node directly loads the checkpoint configured in its YAML, subscribes to ROS,
runs policy inference in-process, and publishes command topics.
Logs are written to --log-dir. Press Ctrl+C to stop both together.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --act-config)
      ACT_CONFIG="${2:-}"
      shift 2
      ;;
    --streaming-act-config)
      STREAMING_ACT_CONFIG="${2:-}"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done
ACT_CONFIG="$(resolve_user_path "${ACT_CONFIG}")"
STREAMING_ACT_CONFIG="$(resolve_user_path "${STREAMING_ACT_CONFIG}")"

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="/tmp/corl_deploy_compare_$(date +%Y%m%d_%H%M%S)"
fi
LOG_DIR="$(resolve_user_path "${LOG_DIR}" true)"
mkdir -p "${LOG_DIR}"

for required_path in \
  "${ACT_CONFIG}" \
  "${STREAMING_ACT_CONFIG}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] Path does not exist: ${required_path}" >&2
    exit 2
  fi
done

start_process() {
  local name="$1"
  local logfile="$2"
  shift 2
  echo "[start] ${name}" | tee -a "${LOG_DIR}/launcher.log"
  (
    cd "${WORKSPACE_ROOT}"
    "$@"
  ) >"${logfile}" 2>&1 &
  local pid=$!
  PIDS+=("${pid}")
  PID_NAMES["${pid}"]="${name}"
  echo "[pid] ${name}=${pid} log=${logfile}" | tee -a "${LOG_DIR}/launcher.log"
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "[stop] Shutting down compare stack..." | tee -a "${LOG_DIR}/launcher.log"
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

declare -a PIDS=()
declare -A PID_NAMES=()

start_process "act_deploy_node" "${LOG_DIR}/act_deploy_node.log" \
  python3 "${DEPLOY_PREFIX}/ros1_adapter/ros1_adapter_node.py" --config "${ACT_CONFIG}"

start_process "streaming_act_deploy_node" "${LOG_DIR}/streaming_act_deploy_node.log" \
  python3 "${DEPLOY_PREFIX}/ros1_adapter/ros1_adapter_node.py" --config "${STREAMING_ACT_CONFIG}"

echo "[ready] Compare stack started." | tee -a "${LOG_DIR}/launcher.log"
echo "[ready] Logs: ${LOG_DIR}" | tee -a "${LOG_DIR}/launcher.log"
echo "[ready] Press Ctrl+C to stop both deploy nodes together." | tee -a "${LOG_DIR}/launcher.log"

set +e
while true; do
  wait -n "${PIDS[@]}"
  wait_status=$?
  exited_pid=""
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      exited_pid="${pid}"
      break
    fi
  done
  if [[ -n "${exited_pid}" ]]; then
    echo "[exit] ${PID_NAMES[${exited_pid}]} exited with status ${wait_status}." | tee -a "${LOG_DIR}/launcher.log"
    exit "${wait_status}"
  fi
done
