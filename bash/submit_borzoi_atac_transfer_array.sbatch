#!/usr/bin/env bash
#SBATCH --job-name=borzoi-atac-xfer
#SBATCH --array=0-31
#SBATCH --partition=akundaje
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:A100_SXM4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=/oak/stanford/groups/akundaje/abuen/ag/outputs/slurm_logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/akundaje/abuen/ag/outputs/slurm_logs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT:-${1:-/oak/stanford/groups/akundaje/abuen/ag/borzoi_transfer_data}}}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/run_borzoi_atac_transfer.sh}"
TRUNK_ROOT="${TRUNK_ROOT:-${DATA_ROOT}/data/borzoi/pretrain_trunks}"
TRUNK_GLOB="${TRUNK_GLOB:-trunk_r*.h5}"
WORK_ROOT="${WORK_ROOT:-${DATA_ROOT}/outputs/borzoi_atac_transfer}"
LOG_ROOT="${LOG_ROOT:-${DATA_ROOT}/outputs/slurm_logs}"

BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS_MIN="${EPOCHS_MIN:-10}"
EPOCHS_MAX="${EPOCHS_MAX:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-20000}"
PROCESSES="${PROCESSES:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-0}"

samples=(${SAMPLES:-K562 GM12878})
modes=(${MODES:-lora})

mkdir -p "${LOG_ROOT}" "${WORK_ROOT}"

if [[ -n "${BORZOI_TRUNKS:-}" ]]; then
  read -r -a trunks <<< "${BORZOI_TRUNKS}"
else
  shopt -s nullglob
  trunks=("${TRUNK_ROOT}"/${TRUNK_GLOB})
  shopt -u nullglob
fi

if (( ${#trunks[@]} == 0 )); then
  echo "No trunk weights found. Set TRUNK_ROOT or BORZOI_TRUNKS." >&2
  exit 1
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
num_samples=${#samples[@]}
num_modes=${#modes[@]}
num_trunks=${#trunks[@]}
total_tasks=$((num_samples * num_modes * num_trunks))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  array_max=$((total_tasks - 1))
  echo "Submitting ${total_tasks} tasks as a Slurm array: 0-${array_max}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "Logs: ${LOG_ROOT}"
  exec sbatch \
    --partition="${PARTITION:-akundaje}" \
    --gres="${GRES:-gpu:1}" \
    --constraint="${CONSTRAINT:-GPU_SKU:A100_SXM4}" \
    --cpus-per-task="${CPUS_PER_TASK:-8}" \
    --mem="${MEM:-96G}" \
    --time="${TIME:-7-00:00:00}" \
    --array="0-${array_max}" \
    --output="${LOG_ROOT}/%x_%A_%a.out" \
    --error="${LOG_ROOT}/%x_%A_%a.err" \
    --export=ALL,DATA_ROOT="${DATA_ROOT}",RUNNER="${RUNNER}",TRUNK_ROOT="${TRUNK_ROOT}",TRUNK_GLOB="${TRUNK_GLOB}",WORK_ROOT="${WORK_ROOT}",LOG_ROOT="${LOG_ROOT}",BATCH_SIZE="${BATCH_SIZE}",EPOCHS_MIN="${EPOCHS_MIN}",EPOCHS_MAX="${EPOCHS_MAX}",WARMUP_STEPS="${WARMUP_STEPS}",PROCESSES="${PROCESSES}",MIXED_PRECISION="${MIXED_PRECISION}",SAMPLES="${SAMPLES:-K562 GM12878}",MODES="${MODES:-lora}" \
    "${BASH_SOURCE[0]}"
fi

if (( task_id >= total_tasks )); then
  echo "Task ${task_id} is outside total_tasks=${total_tasks}; exiting."
  exit 0
fi

trunk_idx=$((task_id % num_trunks))
mode_idx=$(((task_id / num_trunks) % num_modes))
sample_idx=$((task_id / (num_trunks * num_modes)))

sample="${samples[${sample_idx}]}"
mode="${modes[${mode_idx}]}"
trunk="${trunks[${trunk_idx}]}"
trunk_label="$(basename "${trunk}" .h5)"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "sample=${sample}"
echo "mode=${mode}"
echo "trunk=${trunk}"
echo "batch_size=${BATCH_SIZE}"

export DATA_ROOT="${DATA_ROOT}"
export SAMPLES="${sample}"
export MODES="${mode}"
export BORZOI_TRUNKS="${trunk}"
export TRUNK_ROOT="${TRUNK_ROOT}"
export TRUNK_GLOB="${TRUNK_GLOB}"
export WORK_ROOT="${WORK_ROOT}"
export BATCH_SIZE="${BATCH_SIZE}"
export EPOCHS_MIN="${EPOCHS_MIN}"
export EPOCHS_MAX="${EPOCHS_MAX}"
export WARMUP_STEPS="${WARMUP_STEPS}"
export PROCESSES="${PROCESSES}"
export MIXED_PRECISION="${MIXED_PRECISION}"

bash "${RUNNER}" "${DATA_ROOT}"
