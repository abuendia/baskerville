#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/oak/stanford/groups/akundaje/abuen/ag}"
RUNNER="${RUNNER:-${ROOT}/baskerville/bash/run_borzoi_atac_transfer.sh}"
TRUNK_ROOT="${TRUNK_ROOT:-${ROOT}/ag-data/borzoi/pretrain_trunks}"
TRUNK_GLOB="${TRUNK_GLOB:-trunk_r*.h5}"
WORK_ROOT="${WORK_ROOT:-${ROOT}/outputs/borzoi_atac_transfer}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/outputs/local_queue_logs}"

BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS_MIN="${EPOCHS_MIN:-10}"
EPOCHS_MAX="${EPOCHS_MAX:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-20000}"
PROCESSES="${PROCESSES:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
GPUS=(${GPUS:-0 1})

samples=(K562 GM12878)
modes=(${MODES:-lora})

mkdir -p "${LOG_ROOT}" "${WORK_ROOT}"

if (( ${#GPUS[@]} == 0 )); then
  echo "No GPUs configured. Set GPUS, for example: GPUS=\"0 1\"." >&2
  exit 1
fi

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

num_samples=${#samples[@]}
num_modes=${#modes[@]}
num_trunks=${#trunks[@]}
total_tasks=$((num_samples * num_modes * num_trunks))

run_task() {
  local task_id="$1"
  local gpu="$2"
  local trunk_idx mode_idx sample_idx sample mode trunk trunk_label log_base

  trunk_idx=$((task_id % num_trunks))
  mode_idx=$(((task_id / num_trunks) % num_modes))
  sample_idx=$((task_id / (num_trunks * num_modes)))

  sample="${samples[${sample_idx}]}"
  mode="${modes[${mode_idx}]}"
  trunk="${trunks[${trunk_idx}]}"
  trunk_label="$(basename "${trunk}" .h5)"
  log_base="${LOG_ROOT}/task_${task_id}_${sample}_${mode}_${trunk_label}_gpu${gpu}"

  {
    echo "task_id=${task_id}"
    echo "gpu=${gpu}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "sample=${sample}"
    echo "mode=${mode}"
    echo "trunk=${trunk}"
    echo "batch_size=${BATCH_SIZE}"
    echo "mixed_precision=${MIXED_PRECISION}"
    echo "run_eval=${RUN_EVAL}"
    echo "started=$(date -Is)"
  } > "${log_base}.out"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
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
    export RUN_EVAL="${RUN_EVAL}"

    bash "${RUNNER}"
  ) >> "${log_base}.out" 2> "${log_base}.err"

  echo "finished=$(date -Is)" >> "${log_base}.out"
}

QUEUE_STATE="${LOG_ROOT}/queue_state.txt"
QUEUE_LOCK="${LOG_ROOT}/queue.lock"
printf '0\n' > "${QUEUE_STATE}"

echo "Launching ${total_tasks} tasks across GPUs: ${GPUS[*]}"
echo "Logs: ${LOG_ROOT}"
echo "RUN_EVAL=${RUN_EVAL}"

pids=()
for gpu in "${GPUS[@]}"; do
  (
    while true; do
      {
        flock -x 9
        NEXT_TASK="$(<"${QUEUE_STATE}")"
        if (( NEXT_TASK >= total_tasks )); then
          exit 0
        fi
        task_id="${NEXT_TASK}"
        printf '%s\n' "$((NEXT_TASK + 1))" > "${QUEUE_STATE}"
      } 9>"${QUEUE_LOCK}"

      echo "GPU ${gpu} starting task ${task_id}/${total_tasks}"
      run_task "${task_id}" "${gpu}"
      echo "GPU ${gpu} finished task ${task_id}/${total_tasks}"
    done
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
