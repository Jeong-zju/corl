#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

HF_TOKEN=""
HF_USERNAME="${HF_USERNAME:-jeong-zju}"
WANDB_TOKEN="${WANDB_API_KEY:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DOWNLOAD_TOOL="aria2c"
DOWNLOAD_THREADS=4
DOWNLOAD_JOBS=5
POLICY="streaming_act"
INSTALL_SYSTEM_DEPS=1
INSTALL_PYTHON_DEPS=1
DOWNLOAD_DATASETS=1
PROCESS_DATASETS=1
RUN_TRAINING=1
UPLOAD_WATCH=1
DRY_RUN=0
UPLOAD_START_TIMEOUT=900
LOG_ROOT=""
DATASETS=(
  "zeno-ai/CleanTableTopDelayedToolChoice"
  "zeno-ai/BookOriginRelocation"
)
CUSTOM_DATASETS=0
TRAIN_EXTRA_ARGS=()
CHILD_PIDS=()
STARTED_BG_PID=""

usage() {
  cat <<'EOF'
Usage:
  bash bash/install_deploy_zeno.sh --hf-token <hf_token> --wandb-token <wandb_token> [options] [-- extra train args]

One-command setup for the README deployment flow:
  - install system tools used by dataset download
  - install Python requirements and local signatory
  - download the default zeno-ai datasets from Hugging Face
  - build signature caches with data/process_dataset.py
  - launch training and checkpoint upload watchers

Required:
  --hf-token TOKEN              Hugging Face token used for dataset download and uploads.
  --wandb-token TOKEN           W&B API key used for `wandb login` and online training.
                                Required when training is enabled. Can also be
                                provided with WANDB_API_KEY.

Options:
  --hf-username USER            Hugging Face username for gated dataset checks.
                                Default: jeong-zju
  --dataset REPO_ID             Dataset repo id to handle. Can be repeated.
                                Defaults to the two zeno-ai datasets in README.
  --policy POLICY               Policy name passed to bash/train_policy.sh.
                                Default: streaming_act
  --python-bin PATH             Python executable. Default: python3
  --download-tool aria2c|wget   Downloader for data/hfd.sh. Default: aria2c
  -x, --download-threads N      aria2c threads. Default: 4
  -j, --download-jobs N         aria2c concurrent downloads. Default: 5
  --upload-start-timeout SEC    Seconds to wait for a run dir before starting
                                the upload watcher. Default: 900
  --log-root DIR                Directory for train/upload logs.
                                Default: outputs/deploy_logs/<timestamp>
  --skip-system-deps            Do not apt-install curl/jq/ffmpeg/downloader.
  --skip-python-deps            Do not pip-install requirements/signatory.
  --skip-download               Do not download datasets.
  --skip-process                Do not process datasets.
  --skip-train                  Do not launch training.
  --skip-upload-watch           Do not upload checkpoints while training.
  --install-only                Only install system/Python dependencies.
  --data-only                   Install, download, and process data; skip train/upload.
  --dry-run                     Print commands without running them.
  -h, --help                    Show this help.

Examples:
  bash bash/install_deploy_zeno.sh --hf-token hf_xxx --wandb-token wandb_xxx
  bash bash/install_deploy_zeno.sh --hf-token hf_xxx --skip-train
  bash bash/install_deploy_zeno.sh --hf-token hf_xxx --wandb-token wandb_xxx --dataset zeno-ai/BookOriginRelocation -- --steps 1000
EOF
}

log() {
  printf "%b[INFO]%b %s\n" "${GREEN}" "${NC}" "$*"
}

warn() {
  printf "%b[WARN]%b %s\n" "${YELLOW}" "${NC}" "$*" >&2
}

die() {
  printf "%b[ERROR]%b %s\n" "${RED}" "${NC}" "$*" >&2
  exit 1
}

require_value() {
  local flag="$1"
  local value="${2:-}"
  [[ -n "${value}" && "${value}" != --* ]] || die "${flag} requires a value."
}

validate_positive_int() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer."
}

sanitize_name() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  printf "%s" "${value}"
}

format_cmd() {
  local display=("$@")
  local i
  for ((i = 0; i < ${#display[@]}; i++)); do
    if [[ -n "${HF_TOKEN}" && "${display[$i]}" == "${HF_TOKEN}" ]]; then
      display[$i]="<hf-token>"
    elif [[ -n "${WANDB_TOKEN}" && "${display[$i]}" == "${WANDB_TOKEN}" ]]; then
      display[$i]="<wandb-token>"
    fi
  done
  printf "%q " "${display[@]}"
}

run_cmd() {
  log "Running: $(format_cmd "$@")"
  if ((DRY_RUN)); then
    return 0
  fi
  "$@"
}

cleanup_children() {
  local status=$?
  if ((status != 0)); then
    local pid
    for pid in "${CHILD_PIDS[@]:-}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        warn "Stopping background process ${pid}."
        kill "${pid}" 2>/dev/null || true
      fi
    done
  fi
}

trap cleanup_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --hf-token)
        require_value "$1" "${2:-}"
        HF_TOKEN="$2"
        shift 2
        ;;
      --hf-token=*)
        HF_TOKEN="${1#*=}"
        shift
        ;;
      --hf-username)
        require_value "$1" "${2:-}"
        HF_USERNAME="$2"
        shift 2
        ;;
      --hf-username=*)
        HF_USERNAME="${1#*=}"
        shift
        ;;
      --wandb-token)
        require_value "$1" "${2:-}"
        WANDB_TOKEN="$2"
        shift 2
        ;;
      --wandb-token=*)
        WANDB_TOKEN="${1#*=}"
        shift
        ;;
      --dataset)
        require_value "$1" "${2:-}"
        if ((CUSTOM_DATASETS == 0)); then
          DATASETS=()
          CUSTOM_DATASETS=1
        fi
        DATASETS+=("$2")
        shift 2
        ;;
      --dataset=*)
        if ((CUSTOM_DATASETS == 0)); then
          DATASETS=()
          CUSTOM_DATASETS=1
        fi
        DATASETS+=("${1#*=}")
        shift
        ;;
      --policy)
        require_value "$1" "${2:-}"
        POLICY="$2"
        shift 2
        ;;
      --policy=*)
        POLICY="${1#*=}"
        shift
        ;;
      --python-bin)
        require_value "$1" "${2:-}"
        PYTHON_BIN="$2"
        shift 2
        ;;
      --python-bin=*)
        PYTHON_BIN="${1#*=}"
        shift
        ;;
      --download-tool)
        require_value "$1" "${2:-}"
        DOWNLOAD_TOOL="$2"
        shift 2
        ;;
      --download-tool=*)
        DOWNLOAD_TOOL="${1#*=}"
        shift
        ;;
      -x|--download-threads)
        require_value "$1" "${2:-}"
        DOWNLOAD_THREADS="$2"
        shift 2
        ;;
      --download-threads=*)
        DOWNLOAD_THREADS="${1#*=}"
        shift
        ;;
      -j|--download-jobs)
        require_value "$1" "${2:-}"
        DOWNLOAD_JOBS="$2"
        shift 2
        ;;
      --download-jobs=*)
        DOWNLOAD_JOBS="${1#*=}"
        shift
        ;;
      --upload-start-timeout)
        require_value "$1" "${2:-}"
        UPLOAD_START_TIMEOUT="$2"
        shift 2
        ;;
      --upload-start-timeout=*)
        UPLOAD_START_TIMEOUT="${1#*=}"
        shift
        ;;
      --log-root)
        require_value "$1" "${2:-}"
        LOG_ROOT="$2"
        shift 2
        ;;
      --log-root=*)
        LOG_ROOT="${1#*=}"
        shift
        ;;
      --skip-system-deps)
        INSTALL_SYSTEM_DEPS=0
        shift
        ;;
      --skip-python-deps)
        INSTALL_PYTHON_DEPS=0
        shift
        ;;
      --skip-download)
        DOWNLOAD_DATASETS=0
        shift
        ;;
      --skip-process)
        PROCESS_DATASETS=0
        shift
        ;;
      --skip-train)
        RUN_TRAINING=0
        UPLOAD_WATCH=0
        shift
        ;;
      --skip-upload-watch)
        UPLOAD_WATCH=0
        shift
        ;;
      --install-only)
        DOWNLOAD_DATASETS=0
        PROCESS_DATASETS=0
        RUN_TRAINING=0
        UPLOAD_WATCH=0
        shift
        ;;
      --data-only)
        RUN_TRAINING=0
        UPLOAD_WATCH=0
        shift
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        TRAIN_EXTRA_ARGS+=("$@")
        break
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

validate_args() {
  [[ -n "${HF_TOKEN}" ]] || die "Pass the Hugging Face token with --hf-token <token>."
  if ((RUN_TRAINING)) && [[ -z "${WANDB_TOKEN}" ]]; then
    die "Pass the W&B token with --wandb-token <token> or set WANDB_API_KEY."
  fi
  [[ "${DOWNLOAD_TOOL}" == "aria2c" || "${DOWNLOAD_TOOL}" == "wget" ]] || \
    die "--download-tool must be aria2c or wget."
  validate_positive_int "--download-threads" "${DOWNLOAD_THREADS}"
  validate_positive_int "--download-jobs" "${DOWNLOAD_JOBS}"
  validate_positive_int "--upload-start-timeout" "${UPLOAD_START_TIMEOUT}"
  ((${#DATASETS[@]} > 0)) || die "At least one --dataset is required."
}

apt_install_system_deps() {
  local packages=(curl jq ffmpeg)
  if [[ "${DOWNLOAD_TOOL}" == "aria2c" ]]; then
    packages+=(aria2)
  else
    packages+=(wget)
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    warn "apt-get was not found; skipping system package installation."
    return 0
  fi

  local apt_prefix=()
  if [[ "${EUID}" -ne 0 ]]; then
    command -v sudo >/dev/null 2>&1 || die "sudo is required to install system packages."
    apt_prefix=(sudo)
  fi

  run_cmd "${apt_prefix[@]}" apt-get update
  run_cmd "${apt_prefix[@]}" apt-get install -y "${packages[@]}"
}

check_required_commands() {
  local download_command="${DOWNLOAD_TOOL}"
  local commands=(curl "${download_command}" "${PYTHON_BIN}")
  local cmd
  for cmd in "${commands[@]}"; do
    command -v "${cmd}" >/dev/null 2>&1 || die "${cmd} is not installed or not on PATH."
  done

  if ! command -v ffmpeg >/dev/null 2>&1; then
    warn "ffmpeg was not found. torchcodec video decoding may fail without it."
  fi
}

install_python_deps() {
  run_cmd "${PYTHON_BIN}" -m pip install -r requirements.txt
  run_cmd "${PYTHON_BIN}" -m pip install -e depends/signatory --no-build-isolation
}

login_wandb() {
  ((RUN_TRAINING)) || return 0

  if ((DRY_RUN)); then
    run_cmd wandb login --relogin "${WANDB_TOKEN}"
    return 0
  fi

  if ! command -v wandb >/dev/null 2>&1; then
    die "wandb is not installed or not on PATH. Install Python deps or remove --skip-python-deps."
  fi

  run_cmd wandb login --relogin "${WANDB_TOKEN}"
}

configure_tokens() {
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

  if [[ -n "${WANDB_TOKEN}" ]]; then
    export WANDB_API_KEY="${WANDB_TOKEN}"
  fi
}

download_dataset() {
  local dataset="$1"
  local local_dir="data/${dataset}"
  run_cmd \
    bash data/hfd.sh "${dataset}" \
    --dataset \
    --local-dir "${local_dir}" \
    --hf_username "${HF_USERNAME}" \
    --hf_token "${HF_TOKEN}" \
    --tool "${DOWNLOAD_TOOL}" \
    -x "${DOWNLOAD_THREADS}" \
    -j "${DOWNLOAD_JOBS}"
}

process_dataset() {
  local dataset="$1"
  run_cmd "${PYTHON_BIN}" data/process_dataset.py "${dataset}"
}

resolve_train_output_root() {
  local dataset="$1"
  "${PYTHON_BIN}" - "${PROJECT_ROOT}" "${dataset}" "${POLICY}" <<'PY'
from pathlib import Path
import sys
import yaml

project_root = Path(sys.argv[1]).resolve()
dataset = sys.argv[2]
policy = sys.argv[3]

sys.path.insert(0, str(project_root / "scripts"))
from policy_defaults import resolve_cli_dataset_defaults_path

fallback = Path("outputs/train") / dataset / policy.replace("_", "-")
defaults_path = resolve_cli_dataset_defaults_path(
    dataset_selector=dataset,
    task_selector=None,
    policy_name=policy,
)
if defaults_path is None:
    print(fallback.as_posix())
    raise SystemExit(0)

data = yaml.safe_load(Path(defaults_path).read_text(encoding="utf-8")) or {}
train_cfg = data.get("train", {})
if not isinstance(train_cfg, dict):
    train_cfg = {}
print(str(train_cfg.get("output_root") or fallback.as_posix()))
PY
}

absolute_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf "%s" "${path}"
  else
    printf "%s/%s" "${PROJECT_ROOT}" "${path}"
  fi
}

upload_repo_id_for_dataset() {
  local dataset="$1"
  local policy_slug="${POLICY//_/-}"
  printf "%s-%s" "${dataset}" "${policy_slug}"
}

start_background() {
  local log_file="$1"
  shift
  STARTED_BG_PID=""
  log "Starting background command: $(format_cmd "$@")"
  log "Log file: ${log_file}"
  if ((DRY_RUN)); then
    return 0
  fi
  (
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
    [[ -n "${WANDB_TOKEN}" ]] && export WANDB_API_KEY="${WANDB_TOKEN}"
    "$@" >"${log_file}" 2>&1
  ) &
  local pid=$!
  CHILD_PIDS+=("${pid}")
  STARTED_BG_PID="${pid}"
}

wait_for_run_dir() {
  local run_dir="$1"
  local train_pid="$2"
  local timeout="$3"
  local elapsed=0
  while [[ ! -d "${run_dir}" ]]; do
    if ! kill -0 "${train_pid}" 2>/dev/null; then
      return 1
    fi
    if ((elapsed >= timeout)); then
      return 2
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  return 0
}

stop_background_process() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

run_final_checkpoint_upload() {
  local run_dir="$1"
  local repo_id="$2"
  run_cmd \
    "${PYTHON_BIN}" scripts/upload_checkpoints_to_hf.py \
    --run-dir "${run_dir}" \
    --repo-id "${repo_id}" \
    --mode full
}

train_dataset() {
  local dataset="$1"
  local run_stamp
  run_stamp="$(date +%Y%m%d_%H%M%S)"

  local output_root
  output_root="$(resolve_train_output_root "${dataset}")"
  local output_root_abs
  output_root_abs="$(absolute_path "${output_root}")"
  local run_dir="${output_root_abs}/${run_stamp}"
  local safe_dataset
  safe_dataset="$(sanitize_name "${dataset}")"
  local repo_id
  repo_id="$(upload_repo_id_for_dataset "${dataset}")"
  local train_log="${LOG_ROOT}/train_${safe_dataset}_${run_stamp}.log"
  local upload_log="${LOG_ROOT}/upload_${safe_dataset}_${run_stamp}.log"

  log "Dataset: ${dataset}"
  log "Training output: ${run_dir}"
  log "Upload repo: ${repo_id}"

  if ((DRY_RUN)); then
    run_cmd env "CORL_TRAIN_RUN_STAMP=${run_stamp}" \
      bash bash/train_policy.sh --dataset "${dataset}" --policy "${POLICY}" \
      "${TRAIN_EXTRA_ARGS[@]}"
    if ((UPLOAD_WATCH)); then
      run_cmd "${PYTHON_BIN}" scripts/upload_checkpoints_to_hf.py \
        --run-dir "${run_dir}" --repo-id "${repo_id}" --mode full --watch
    fi
    return 0
  fi

  mkdir -p "${LOG_ROOT}"
  (
    export CORL_TRAIN_RUN_STAMP="${run_stamp}"
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
    [[ -n "${WANDB_TOKEN}" ]] && export WANDB_API_KEY="${WANDB_TOKEN}"
    bash bash/train_policy.sh --dataset "${dataset}" --policy "${POLICY}" \
      "${TRAIN_EXTRA_ARGS[@]}" >"${train_log}" 2>&1
  ) &
  local train_pid=$!
  CHILD_PIDS+=("${train_pid}")
  log "Training PID: ${train_pid}"
  log "Training log: ${train_log}"

  local upload_pid=""
  if ((UPLOAD_WATCH)); then
    local wait_status=0
    wait_for_run_dir "${run_dir}" "${train_pid}" "${UPLOAD_START_TIMEOUT}" || wait_status=$?
    if ((wait_status == 0)); then
      start_background "${upload_log}" \
        "${PYTHON_BIN}" scripts/upload_checkpoints_to_hf.py \
        --run-dir "${run_dir}" \
        --repo-id "${repo_id}" \
        --mode full \
        --watch
      upload_pid="${STARTED_BG_PID}"
      log "Upload watcher PID: ${upload_pid}"
    elif ((wait_status == 2)); then
      warn "Timed out waiting for ${run_dir}; checkpoint upload watcher was not started."
    else
      warn "Training exited before ${run_dir} was created; skipping upload watcher."
    fi
  fi

  local train_status=0
  set +e
  wait "${train_pid}"
  train_status=$?
  set -e

  if [[ -n "${upload_pid}" ]]; then
    stop_background_process "${upload_pid}"
  fi

  if ((train_status != 0)); then
    warn "Training failed for ${dataset}. See ${train_log}."
    return "${train_status}"
  fi

  log "Training finished for ${dataset}."
  if ((UPLOAD_WATCH)); then
    run_final_checkpoint_upload "${run_dir}" "${repo_id}"
  fi
}

main() {
  parse_args "$@"
  validate_args
  configure_tokens

  if [[ -z "${LOG_ROOT}" ]]; then
    LOG_ROOT="outputs/deploy_logs/$(date +%Y%m%d_%H%M%S)"
  fi

  log "Project root: ${PROJECT_ROOT}"
  log "Datasets: ${DATASETS[*]}"
  log "Policy: ${POLICY}"

  if ((INSTALL_SYSTEM_DEPS)); then
    apt_install_system_deps
  fi
  check_required_commands

  if ((INSTALL_PYTHON_DEPS)); then
    install_python_deps
  fi

  login_wandb

  if ((DOWNLOAD_DATASETS)); then
    local dataset
    for dataset in "${DATASETS[@]}"; do
      download_dataset "${dataset}"
    done
  fi

  if ((PROCESS_DATASETS)); then
    local dataset
    for dataset in "${DATASETS[@]}"; do
      process_dataset "${dataset}"
    done
  fi

  if ((RUN_TRAINING)); then
    mkdir -p "${LOG_ROOT}"
    local dataset
    for dataset in "${DATASETS[@]}"; do
      train_dataset "${dataset}"
    done
  fi

  log "Done."
}

main "$@"
