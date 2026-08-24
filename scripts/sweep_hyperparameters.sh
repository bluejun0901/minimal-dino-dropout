#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Create it with: uv sync --extra dev" >&2
  exit 1
fi
source .venv/bin/activate

SWEEP="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ "${SWEEP}" != "all" && "${SWEEP}" != "dropout" \
  && "${SWEEP}" != "center-momentum" && "${SWEEP}" != "teacher-momentum" ]]; then
  echo "Usage: $0 [all|dropout|center-momentum|teacher-momentum] [extra train args...]" >&2
  exit 2
fi

# Edit these grids to change the sweep. Each run changes only the named value;
# all other hyperparameters remain at the current baseline below.
DROPOUT_VALUES=(0.0 0.05 0.1 0.2 0.3)
CENTER_MOMENTUM_VALUES=(0.5 0.9 0.99 0.999)
TEACHER_MOMENTUM_VALUES=(0.9 0.99 0.996 0.999)

TRAIN_FILE="${TRAIN_FILE:-data/wiki1m_for_simcse.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/sweeps}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Training file not found: ${TRAIN_FILE}" >&2
  exit 1
fi

COMMON_ARGS=(
  --train-file "${TRAIN_FILE}"
  --model-name bert-base-uncased
  --model-revision 86b5e0934494bd15c9632b12f734a8a67f723594
  --epochs 1
  --batch-size 64
  --max-length 32
  --learning-rate 3e-5
  --seed 42
  --log-steps 10
  --eval-steps 250
  --save-steps 500
  --keep-last-checkpoints 2
  --device "${DEVICE}"
  "$@"
)

run_experiment() {
  local sweep_name="$1"
  local value="$2"
  local varied_flag="$3"
  local output_dir="${OUTPUT_ROOT}/${sweep_name}/${sweep_name}-${value}"

  if [[ -e "${output_dir}" ]]; then
    echo "Output already exists, refusing to mix runs: ${output_dir}" >&2
    echo "Set OUTPUT_ROOT to a new directory or remove/move the existing run." >&2
    exit 1
  fi

  echo "[$(date '+%F %T')] ${sweep_name}=${value} -> ${output_dir}"
  uv run python -m minimal_dino.train \
    "${COMMON_ARGS[@]}" \
    --output-dir "${output_dir}" \
    --dropout 0.1 \
    --center-momentum 0.9 \
    --teacher-momentum 0.996 \
    "${varied_flag}" "${value}"
}

if [[ "${SWEEP}" == "all" || "${SWEEP}" == "dropout" ]]; then
  for value in "${DROPOUT_VALUES[@]}"; do
    run_experiment dropout "${value}" --dropout
  done
fi

if [[ "${SWEEP}" == "all" || "${SWEEP}" == "center-momentum" ]]; then
  for value in "${CENTER_MOMENTUM_VALUES[@]}"; do
    run_experiment center-momentum "${value}" --center-momentum
  done
fi

if [[ "${SWEEP}" == "all" || "${SWEEP}" == "teacher-momentum" ]]; then
  for value in "${TEACHER_MOMENTUM_VALUES[@]}"; do
    run_experiment teacher-momentum "${value}" --teacher-momentum
  done
fi
