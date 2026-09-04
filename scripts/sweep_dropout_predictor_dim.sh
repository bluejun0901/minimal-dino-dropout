#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Create it with: uv sync --extra dev" >&2
  exit 1
fi
source .venv/bin/activate

# Override grids with whitespace-separated environment variables, for example:
#   DROPOUT_GRID="0.1 0.2" PREDICTOR_DIM_GRID="1024 4096" ./scripts/sweep_dropout_predictor_dim.sh
read -r -a dropout_values <<< "${DROPOUT_GRID:-0.05 0.1 0.2 0.3}"
read -r -a predictor_dim_values <<< "${PREDICTOR_DIM_GRID:-512 1024 2048 4096}"

TRAIN_FILE="${TRAIN_FILE:-data/wiki1m_for_simcse.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/dropout-predictor-dim-sweep}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Training file not found: ${TRAIN_FILE}" >&2
  exit 1
fi
if (( ${#dropout_values[@]} == 0 || ${#predictor_dim_values[@]} == 0 )); then
  echo "DROPOUT_GRID and PREDICTOR_DIM_GRID must not be empty" >&2
  exit 2
fi

# Arguments passed to this script are forwarded as Hydra overrides. The sweep
# keys and output directory are appended afterward so every run remains isolated.
extra_args=("$@")

run_dir() {
  local dropout="$1"
  local predictor_dim="$2"
  printf '%s/dropout-%s_predictor-dim-%s' "${OUTPUT_ROOT}" "${dropout}" "${predictor_dim}"
}

# Check the entire grid first so an existing directory cannot stop a sweep only
# after some of its runs have already completed.
for dropout in "${dropout_values[@]}"; do
  for predictor_dim in "${predictor_dim_values[@]}"; do
    output_dir="$(run_dir "${dropout}" "${predictor_dim}")"
    if [[ -e "${output_dir}" ]]; then
      echo "Output already exists, refusing to mix runs: ${output_dir}" >&2
      echo "Set OUTPUT_ROOT to a new directory or move the existing run." >&2
      exit 1
    fi
  done
done

total_runs=$(( ${#dropout_values[@]} * ${#predictor_dim_values[@]} ))
run_number=0

for dropout in "${dropout_values[@]}"; do
  for predictor_dim in "${predictor_dim_values[@]}"; do
    run_number=$((run_number + 1))
    output_dir="$(run_dir "${dropout}" "${predictor_dim}")"

    echo "[$(date '+%F %T')] (${run_number}/${total_runs}) dropout=${dropout}, predictor_dim=${predictor_dim} -> ${output_dir}"
    uv run python -m minimal_dino.train \
      "data.train_file=${TRAIN_FILE}" \
      "runtime.device=${DEVICE}" \
      "${extra_args[@]}" \
      "model.dropout=${dropout}" \
      "model.predictor_hidden_dim=${predictor_dim}" \
      "runtime.output_dir=${output_dir}"
  done
done
