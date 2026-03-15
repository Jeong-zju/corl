#!/usr/bin/env bash

find_latest_run_dir() {
  local train_root="$1"
  local latest=""

  if [[ -d "${train_root}" ]]; then
    latest="$(
      find "${train_root}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )"
  fi

  printf '%s\n' "${latest}"
}

resolve_policy_dir_from_run() {
  local run_dir="$1"

  if [[ -f "${run_dir}/checkpoints/last/pretrained_model/model.safetensors" ]]; then
    printf '%s\n' "${run_dir}/checkpoints/last/pretrained_model"
    return 0
  fi

  if [[ -f "${run_dir}/pretrained_model/model.safetensors" ]]; then
    printf '%s\n' "${run_dir}/pretrained_model"
    return 0
  fi

  if [[ -f "${run_dir}/model.safetensors" ]]; then
    printf '%s\n' "${run_dir}"
    return 0
  fi

  return 1
}

resolve_eval_policy_path() {
  local policy_path="$1"
  local latest_run_dir="$2"
  local train_output_root="$3"

  if [[ -z "${policy_path}" && -z "${latest_run_dir}" ]]; then
    latest_run_dir="$(find_latest_run_dir "${train_output_root}")"
  fi

  if [[ -z "${policy_path}" && -n "${latest_run_dir}" ]]; then
    if ! policy_path="$(resolve_policy_dir_from_run "${latest_run_dir}")"; then
      echo "Latest run does not contain policy weights: ${latest_run_dir}" >&2
      echo "Expected one of:" >&2
      echo "  ${latest_run_dir}/checkpoints/last/pretrained_model/model.safetensors" >&2
      echo "  ${latest_run_dir}/pretrained_model/model.safetensors" >&2
      echo "  ${latest_run_dir}/model.safetensors" >&2
      return 1
    fi
  fi

  if [[ -z "${policy_path}" ]]; then
    echo "Could not infer latest run from TRAIN_OUTPUT_ROOT=${train_output_root}" >&2
    echo "Set POLICY_PATH or LATEST_RUN_DIR explicitly, or train a model first." >&2
    return 1
  fi

  printf '%s\n' "${policy_path}"
}
