#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Create it with: uv sync --extra dev" >&2
  exit 1
fi
source .venv/bin/activate

# Each run starts from the current Hydra defaults and changes one conceptual
# factor: either center_scale, or one regularizer's weight.
CENTER_SCALE_VALUES=(0.05 0.1 0.2 0.5 1)
LOSS_WEIGHTS=(0.05 0.01 0.005 0.001 0.0005)
REGULARIZER_MODES=(normalized_mean koleo covariance)
REGULARIZER_NAMES=(uniformity koleo covariance)
OUTPUT_ROOT="runs/analyze"

run_dirs=()
for value_index in "${!LOSS_WEIGHTS[@]}"; do
  value="${CENTER_SCALE_VALUES[value_index]}"
  weight="${LOSS_WEIGHTS[value_index]}"
  run_dirs+=("${OUTPUT_ROOT}/center_scale_${value}")
  for index in "${!REGULARIZER_MODES[@]}"; do
    run_dirs+=("${OUTPUT_ROOT}/${REGULARIZER_NAMES[index]}_${weight}")
  done
done

# Validate every destination before starting so a collision cannot leave a
# newly started sweep only partially completed.
for output_dir in "${run_dirs[@]}"; do
  if [[ -e "${output_dir}" ]]; then
    echo "Output already exists, refusing to mix runs: ${output_dir}" >&2
    echo "Move or remove existing analyze runs before starting this sweep." >&2
    exit 1
  fi
done

total_runs=${#run_dirs[@]}
run_number=0

echo "[$(date '+%F %T')] Starting sweep of center_scale and regularizers with ${total_runs} runs."

echo "[$(date '+%F %T')] Baseline run (no center_scale, no regularizer) -> ${OUTPUT_ROOT}/baseline"
uv run python -m minimal_dino.train \
  "runtime.output_dir=${OUTPUT_ROOT}/baseline"

for value_index in "${!LOSS_WEIGHTS[@]}"; do
  value="${CENTER_SCALE_VALUES[value_index]}"
  weight="${LOSS_WEIGHTS[value_index]}"

  run_number=$((run_number + 1))
  output_dir="${OUTPUT_ROOT}/center_scale_${value}"
  echo "[$(date '+%F %T')] (${run_number}/${total_runs}) center_scale=${value} -> ${output_dir}"
  uv run python -m minimal_dino.train \
    "objective.center_scale=${value}" \
    "runtime.output_dir=${output_dir}"

  for index in "${!REGULARIZER_MODES[@]}"; do
    mode="${REGULARIZER_MODES[index]}"
    name="${REGULARIZER_NAMES[index]}"
    run_number=$((run_number + 1))
    output_dir="${OUTPUT_ROOT}/${name}_${weight}"
    echo "[$(date '+%F %T')] (${run_number}/${total_runs}) ${name}_weight=${weight} -> ${output_dir}"
    uv run python -m minimal_dino.train \
      "objective.uniformity_mode=${mode}" \
      "objective.uniformity_weight=${weight}" \
      "runtime.output_dir=${output_dir}"
  done
done
