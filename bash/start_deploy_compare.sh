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
    if [[ -f "${candidate}/main/deploy/policy_runtime/server.py" ]]; then
      echo "${candidate}"
      return 0
    fi
    if [[ -f "${candidate}/deploy/policy_runtime/server.py" ]]; then
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

ACT_POLICY_PATH=""
STREAMING_ACT_POLICY_PATH=""
ACT_DEVICE="cuda"
STREAMING_ACT_DEVICE="cuda"
ACT_LOAD_DEVICE=""
STREAMING_ACT_LOAD_DEVICE=""
ACT_N_ACTION_STEPS=""
STREAMING_ACT_N_ACTION_STEPS=""
ACT_CONFIG="${DEPLOY_PREFIX}/configs/deploy_zeno_compare_act.yaml"
STREAMING_ACT_CONFIG="${DEPLOY_PREFIX}/configs/deploy_zeno_compare_streaming_act.yaml"
LOG_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash main/bash/start_deploy_compare.sh \
    --act-policy-path <act_ckpt_dir> \
    --streaming-act-policy-path <streaming_act_ckpt_dir> \
    [--act-device cuda] \
    [--streaming-act-device cuda] \
    [--act-load-device cpu] \
    [--streaming-act-load-device cpu] \
    [--act-n-action-steps 1] \
    [--streaming-act-n-action-steps 1] \
    [--act-config <compare_act_yaml>] \
    [--streaming-act-config <compare_streaming_act_yaml>] \
    [--log-dir /tmp/corl_deploy_compare]

This script starts 6 processes:
  1. ACT policy runtime
  2. ACT bridge
  3. ACT ROS1 adapter
  4. Streaming ACT policy runtime
  5. Streaming ACT bridge
  6. Streaming ACT ROS1 adapter

Logs are written to --log-dir. Press Ctrl+C to stop everything together.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --act-policy-path)
      ACT_POLICY_PATH="${2:-}"
      shift 2
      ;;
    --streaming-act-policy-path)
      STREAMING_ACT_POLICY_PATH="${2:-}"
      shift 2
      ;;
    --act-device)
      ACT_DEVICE="${2:-}"
      shift 2
      ;;
    --streaming-act-device)
      STREAMING_ACT_DEVICE="${2:-}"
      shift 2
      ;;
    --act-load-device)
      ACT_LOAD_DEVICE="${2:-}"
      shift 2
      ;;
    --streaming-act-load-device)
      STREAMING_ACT_LOAD_DEVICE="${2:-}"
      shift 2
      ;;
    --act-n-action-steps)
      ACT_N_ACTION_STEPS="${2:-}"
      shift 2
      ;;
    --streaming-act-n-action-steps)
      STREAMING_ACT_N_ACTION_STEPS="${2:-}"
      shift 2
      ;;
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

if [[ -z "${ACT_POLICY_PATH}" || -z "${STREAMING_ACT_POLICY_PATH}" ]]; then
  echo "[ERROR] --act-policy-path and --streaming-act-policy-path are required." >&2
  usage >&2
  exit 2
fi

ACT_POLICY_PATH="$(resolve_user_path "${ACT_POLICY_PATH}")"
STREAMING_ACT_POLICY_PATH="$(resolve_user_path "${STREAMING_ACT_POLICY_PATH}")"
ACT_CONFIG="$(resolve_user_path "${ACT_CONFIG}")"
STREAMING_ACT_CONFIG="$(resolve_user_path "${STREAMING_ACT_CONFIG}")"

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="/tmp/corl_deploy_compare_$(date +%Y%m%d_%H%M%S)"
fi
LOG_DIR="$(resolve_user_path "${LOG_DIR}" true)"
mkdir -p "${LOG_DIR}"

for required_path in \
  "${ACT_POLICY_PATH}" \
  "${STREAMING_ACT_POLICY_PATH}" \
  "${ACT_CONFIG}" \
  "${STREAMING_ACT_CONFIG}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] Path does not exist: ${required_path}" >&2
    exit 2
  fi
done

read_config_endpoints() {
  local config_path="$1"
  python3 - "$config_path" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
bridge = data.get("bridge", {})
policy_endpoint = str(bridge.get("policy_endpoint", "")).strip()
policy_control_endpoint = str(bridge.get("policy_control_endpoint", "")).strip()
if not policy_endpoint:
    raise SystemExit("Missing bridge.policy_endpoint in config: " + str(path))
print(policy_endpoint)
print(policy_control_endpoint)
PY
}

endpoint_to_bind() {
  local endpoint="$1"
  python3 - "$endpoint" <<'PY'
import sys

endpoint = sys.argv[1].strip()
if not endpoint:
    print("")
    raise SystemExit(0)
if not endpoint.startswith("tcp://") or ":" not in endpoint.rsplit("/", 1)[-1]:
    raise SystemExit("Unsupported endpoint format: " + endpoint)
port = endpoint.rsplit(":", 1)[1]
print(f"tcp://*:{port}")
PY
}

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

mapfile -t ACT_ENDPOINTS < <(read_config_endpoints "${ACT_CONFIG}")
mapfile -t STREAMING_ENDPOINTS < <(read_config_endpoints "${STREAMING_ACT_CONFIG}")

ACT_POLICY_BIND="$(endpoint_to_bind "${ACT_ENDPOINTS[0]}")"
ACT_POLICY_CONTROL_BIND="$(endpoint_to_bind "${ACT_ENDPOINTS[1]}")"
STREAMING_POLICY_BIND="$(endpoint_to_bind "${STREAMING_ENDPOINTS[0]}")"
STREAMING_POLICY_CONTROL_BIND="$(endpoint_to_bind "${STREAMING_ENDPOINTS[1]}")"

act_runtime_cmd=(
  python3 "${DEPLOY_PREFIX}/policy_runtime/server.py"
  --policy-type act
  --policy-path "${ACT_POLICY_PATH}"
  --device "${ACT_DEVICE}"
  --bind "${ACT_POLICY_BIND}"
  --control-bind "${ACT_POLICY_CONTROL_BIND}"
)
if [[ -n "${ACT_LOAD_DEVICE}" ]]; then
  act_runtime_cmd+=(--load-device "${ACT_LOAD_DEVICE}")
fi
if [[ -n "${ACT_N_ACTION_STEPS}" ]]; then
  act_runtime_cmd+=(--n-action-steps "${ACT_N_ACTION_STEPS}")
fi

streaming_runtime_cmd=(
  python3 "${DEPLOY_PREFIX}/policy_runtime/server.py"
  --policy-type streaming_act
  --policy-path "${STREAMING_ACT_POLICY_PATH}"
  --device "${STREAMING_ACT_DEVICE}"
  --bind "${STREAMING_POLICY_BIND}"
  --control-bind "${STREAMING_POLICY_CONTROL_BIND}"
)
if [[ -n "${STREAMING_ACT_LOAD_DEVICE}" ]]; then
  streaming_runtime_cmd+=(--load-device "${STREAMING_ACT_LOAD_DEVICE}")
fi
if [[ -n "${STREAMING_ACT_N_ACTION_STEPS}" ]]; then
  streaming_runtime_cmd+=(--n-action-steps "${STREAMING_ACT_N_ACTION_STEPS}")
fi

start_process "act_policy_runtime" "${LOG_DIR}/act_policy_runtime.log" "${act_runtime_cmd[@]}"
start_process "act_bridge" "${LOG_DIR}/act_bridge.log" \
  python3 "${DEPLOY_PREFIX}/bridge/bridge_core.py" --config "${ACT_CONFIG}"
start_process "act_ros1_adapter" "${LOG_DIR}/act_ros1_adapter.log" \
  python3 "${DEPLOY_PREFIX}/ros1_adapter/ros1_adapter_node.py" --config "${ACT_CONFIG}"

start_process "streaming_act_policy_runtime" "${LOG_DIR}/streaming_act_policy_runtime.log" "${streaming_runtime_cmd[@]}"
start_process "streaming_act_bridge" "${LOG_DIR}/streaming_act_bridge.log" \
  python3 "${DEPLOY_PREFIX}/bridge/bridge_core.py" --config "${STREAMING_ACT_CONFIG}"
start_process "streaming_act_ros1_adapter" "${LOG_DIR}/streaming_act_ros1_adapter.log" \
  python3 "${DEPLOY_PREFIX}/ros1_adapter/ros1_adapter_node.py" --config "${STREAMING_ACT_CONFIG}"

echo "[ready] Compare stack started." | tee -a "${LOG_DIR}/launcher.log"
echo "[ready] Logs: ${LOG_DIR}" | tee -a "${LOG_DIR}/launcher.log"
echo "[ready] Press Ctrl+C to stop all 6 processes together." | tee -a "${LOG_DIR}/launcher.log"

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
