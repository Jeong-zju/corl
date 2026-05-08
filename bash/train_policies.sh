#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS=()
ENVS=()
POLICIES=()
TRAIN_EXTRA_ARGS=()
TARGET_KIND=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash bash/train_policies.sh --dataset <dataset> --policy <policy> [--policy <policy> ...] [-- extra train args]
  bash bash/train_policies.sh --env <env> --policy <policy> [--policy <policy> ...] [-- extra train args]

Description:
  Sequentially runs bash/train_policy.sh once per requested policy.
  Use --dataset for explicit dataset selectors, or --env to let
  bash/train_policy.sh resolve the dataset from the default config.

Options:
  --dataset DATASET        Dataset selector to train on. Can be repeated.
  --env ENV                Environment name to train on. Can be repeated.
  --policy POLICY          Policy name to train. Can be repeated.
  --policies A,B,C         Comma-separated list of policies.
  --dry-run                Print the commands without running them.
  --                        Forward all remaining arguments to every
                           bash/train_policy.sh invocation.
  -h, --help               Show this help.

Examples:
  bash bash/train_policies.sh --dataset zeno-ai/DailyLaundryOrganization --policy act --policy streaming_act
  bash bash/train_policies.sh --env braidedhub --policy streaming_act --policy diffusion
  bash bash/train_policies.sh --dataset zeno-ai/DailyLaundryOrganization --policy act --policy streaming_act -- --steps 20000
EOF
}

log() {
  printf "[INFO] %s\n" "$*"
}

warn() {
  printf "[WARN] %s\n" "$*" >&2
}

die() {
  printf "[ERROR] %s\n" "$*" >&2
  exit 1
}

require_value() {
  local flag="$1"
  local value="${2:-}"
  [[ -n "${value}" && "${value}" != --* ]] || die "${flag} requires a value."
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf "%s" "${value}"
}

format_cmd() {
  local rendered
  rendered="$(printf "%q " "$@")"
  printf "%s" "${rendered% }"
}

append_policies_csv() {
  local csv="$1"
  local item
  local policy_items=()
  IFS=',' read -r -a policy_items <<<"${csv}"
  for item in "${policy_items[@]}"; do
    item="$(trim "${item}")"
    [[ -n "${item}" ]] && POLICIES+=("${item}")
  done
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset)
        require_value "$1" "${2:-}"
        if [[ "${TARGET_KIND}" == "env" ]]; then
          die "Use either --dataset or --env in one invocation, not both."
        fi
        TARGET_KIND="dataset"
        DATASETS+=("$2")
        shift 2
        ;;
      --dataset=*)
        if [[ "${TARGET_KIND}" == "env" ]]; then
          die "Use either --dataset or --env in one invocation, not both."
        fi
        TARGET_KIND="dataset"
        DATASETS+=("${1#*=}")
        shift
        ;;
      --env)
        require_value "$1" "${2:-}"
        if [[ "${TARGET_KIND}" == "dataset" ]]; then
          die "Use either --dataset or --env in one invocation, not both."
        fi
        TARGET_KIND="env"
        ENVS+=("$2")
        shift 2
        ;;
      --env=*)
        if [[ "${TARGET_KIND}" == "dataset" ]]; then
          die "Use either --dataset or --env in one invocation, not both."
        fi
        TARGET_KIND="env"
        ENVS+=("${1#*=}")
        shift
        ;;
      --policy)
        require_value "$1" "${2:-}"
        POLICIES+=("$2")
        shift 2
        ;;
      --policy=*)
        POLICIES+=("${1#*=}")
        shift
        ;;
      --policies)
        require_value "$1" "${2:-}"
        append_policies_csv "$2"
        shift 2
        ;;
      --policies=*)
        append_policies_csv "${1#*=}"
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
  [[ -n "${TARGET_KIND}" ]] || die "Pass at least one --dataset or --env."
  ((${#POLICIES[@]} > 0)) || die "Pass at least one --policy or --policies value."

  if [[ "${TARGET_KIND}" == "dataset" ]]; then
    ((${#DATASETS[@]} > 0)) || die "Pass at least one --dataset value."
  else
    ((${#ENVS[@]} > 0)) || die "Pass at least one --env value."
  fi
}

run_one() {
  local target_kind="$1"
  local target_value="$2"
  local policy="$3"
  local run_index="$4"

  local cmd=(
    bash bash/train_policy.sh
    "--${target_kind}" "${target_value}"
    --policy "${policy}"
  )

  if ((${#TRAIN_EXTRA_ARGS[@]} > 0)); then
    cmd+=("${TRAIN_EXTRA_ARGS[@]}")
  fi

  log "Run ${run_index}: ${target_kind}=${target_value}, policy=${policy}"
  log "Command: $(format_cmd "${cmd[@]}")"

  if ((DRY_RUN)); then
    return 0
  fi

  if ! "${cmd[@]}"; then
    die "Training failed for ${target_kind}=${target_value}, policy=${policy}."
  fi
}

main() {
  parse_args "$@"
  validate_args

  log "Project root: ${REPO_ROOT}"
  log "Mode: ${TARGET_KIND}"
  log "Policies: ${POLICIES[*]}"
  if ((${#TRAIN_EXTRA_ARGS[@]} > 0)); then
    log "Forwarded args: $(format_cmd "${TRAIN_EXTRA_ARGS[@]}")"
  fi
  if ((DRY_RUN)); then
    warn "Dry run enabled; no training command will be executed."
  fi

  local targets=()
  if [[ "${TARGET_KIND}" == "dataset" ]]; then
    targets=("${DATASETS[@]}")
  else
    targets=("${ENVS[@]}")
  fi

  local target
  local policy
  local run_index=0

  for target in "${targets[@]}"; do
    for policy in "${POLICIES[@]}"; do
      ((++run_index))
      run_one "${TARGET_KIND}" "${target}" "${policy}" "${run_index}"
    done
  done
}

main "$@"
