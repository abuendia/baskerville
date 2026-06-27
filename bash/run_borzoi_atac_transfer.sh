#!/usr/bin/env bash
set -euo pipefail

BASKERVILLE_DIR="${BASKERVILLE_DIR:-/mnt/scratch/ag/baskerville}"
CONDA_SH="${CONDA_SH:-/mnt/scratch/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-baskerville}"

WORK_ROOT="${WORK_ROOT:-/mnt/scratch/ag/data/borzoi_atac_transfer}"
FASTA="${FASTA:-/mnt/scratch/ag/data/public_data/genome/hg38.genome.fa}"
TRUNK_ROOT="${TRUNK_ROOT:-/mnt/scratch/ag/data/borzoi/pretrain_trunks}"
BORZOI_TRUNKS="${BORZOI_TRUNKS:-}"
AG_FOLD_DIR="${AG_FOLD_DIR:-/mnt/scratch/ag/data/ag-data/ag_regions/fold_1}"
USE_AG_FOLD="${USE_AG_FOLD:-1}"

SEQ_LENGTH="${SEQ_LENGTH:-524288}"
CROP_BP="${CROP_BP:-163840}"
POOL_WIDTH="${POOL_WIDTH:-32}"
FOLDS="${FOLDS:-8}"
SETUP_FOLD_SUBSET="${SETUP_FOLD_SUBSET:-4}"
TRAIN_FOLD_DIR="${TRAIN_FOLD_DIR:-f3c0}"
SEQS_PER_TFR="${SEQS_PER_TFR:-256}"
PROCESSES="${PROCESSES:-16}"

LIMIT_BED="${LIMIT_BED:-}"
BLACKLIST_BED="${BLACKLIST_BED:-}"
UMAP_BED="${UMAP_BED:-}"

MODES="${MODES:-full linear lora locon}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS_MIN="${EPOCHS_MIN:-1}"
EPOCHS_MAX="${EPOCHS_MAX:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
SCALE="${SCALE:-1.0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
FORCE="${FORCE:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-0}"

export PYTHONPATH="${BASKERVILLE_DIR}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/scratch/tmp/mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/scratch/tmp}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${BASKERVILLE_DIR}"
mkdir -p "${WORK_ROOT}"

declare -A BIGWIGS=(
  [K562]="/mnt/scratch/ag/data/ag-data/atac_ft/K562/K562_ATAC.bw"
  [GM12878]="/mnt/scratch/ag/data/ag-data/atac_ft/GM12878/GM12878_ATAC.bw"
)

if [[ -n "${BORZOI_TRUNKS}" ]]; then
  read -r -a TRUNKS <<< "${BORZOI_TRUNKS}"
else
  shopt -s nullglob
  TRUNKS=("${TRUNK_ROOT}"/trunk_r*.h5)
  shopt -u nullglob
fi

if (( ${#TRUNKS[@]} == 0 )); then
  cat >&2 <<EOF
No Borzoi trunk weights found.

Set one of:
  BORZOI_TRUNKS="/path/to/trunk_r0.h5 /path/to/trunk_r1.h5"
  TRUNK_ROOT=/directory/containing/trunk_r*.h5

The Baskerville transfer tutorial restores trunk weights with:
  hound_transfer.py --trunk --restore <trunk.h5> ...
EOF
  exit 1
fi

run_hound_data() {
  local data_dir="$1"
  local targets_file="${data_dir}/targets.txt"

  if [[ -f "${data_dir}/statistics.json" && "${FORCE}" != "1" ]]; then
    echo "TFRecords already prepared: ${data_dir}"
    return
  fi

  mkdir -p "${data_dir}"
	  local cmd=(
	    python src/baskerville/scripts/hound_data.py
	    --restart
	    --local
	    -c "${CROP_BP}"
	    -d 2
	    -l "${SEQ_LENGTH}"
	    -p "${PROCESSES}"
	    -r "${SEQS_PER_TFR}"
	    -w "${POOL_WIDTH}"
	    -o "${data_dir}"
	  )

	  if [[ "${USE_AG_FOLD}" != "1" ]]; then
	    cmd+=(-f "${FOLDS}")
	  fi

  if [[ -n "${LIMIT_BED}" ]]; then
    cmd+=(--limit "${LIMIT_BED}")
  fi
  if [[ -n "${BLACKLIST_BED}" ]]; then
    cmd+=(-b "${BLACKLIST_BED}")
  fi
  if [[ -n "${UMAP_BED}" ]]; then
    cmd+=(-u "${UMAP_BED}" --umap_clip 0.5)
  fi

  cmd+=("${FASTA}" "${targets_file}")
  echo "Creating TFRecords in ${data_dir}"
  "${cmd[@]}"
}

setup_folds() {
  local sample_dir="$1"
  local params_file="$2"
  local data_dir="$3"
  local folds_dir="${sample_dir}/folds"

  if [[ -d "${folds_dir}/${TRAIN_FOLD_DIR}/data0" && "${FORCE}" != "1" ]]; then
    echo "Fold setup already exists: ${folds_dir}"
    return
  fi

  if [[ -d "${folds_dir}" && "${FORCE}" == "1" ]]; then
    rm -rf "${folds_dir}"
  fi

  python docs/transfer_human/setup_folds.py \
    -o "${folds_dir}" \
    -f "${SETUP_FOLD_SUBSET}" \
    "${params_file}" \
    "${data_dir}"
}

for sample in K562 GM12878; do
  sample_dir="${WORK_ROOT}/${sample}"
  w5_dir="${sample_dir}/w5"
  data_dir="${sample_dir}/tfr"
  params_dir="${sample_dir}/params"
  train_data_dir="${sample_dir}/folds/${TRAIN_FOLD_DIR}/data0"
  bw="${BIGWIGS[${sample}]}"
  w5="${w5_dir}/${sample}_ATAC.w5"

  mkdir -p "${w5_dir}" "${data_dir}" "${params_dir}"

  if [[ ! -f "${w5}" || "${FORCE}" == "1" ]]; then
    echo "Converting ${bw} -> ${w5}"
    python src/baskerville/scripts/utils/bw_w5.py "${bw}" "${w5}"
  else
    echo "W5 already exists: ${w5}"
  fi

	  python scripts/prepare_atac_transfer_configs.py \
	    --data-dir "${data_dir}" \
	    --params-dir "${params_dir}" \
	    --identifier "${sample}_ATAC" \
	    --description "${sample} ATAC" \
	    --w5 "${w5}" \
	    --seq-length "${SEQ_LENGTH}" \
	    --crop-bp "${CROP_BP}" \
	    --batch-size "${BATCH_SIZE}" \
	    --epochs-min "${EPOCHS_MIN}" \
	    --epochs-max "${EPOCHS_MAX}" \
	    --warmup-steps "${WARMUP_STEPS}" \
	    --scale "${SCALE}" \
	    --modes ${MODES} \
	    $(if [[ "${USE_AG_FOLD}" == "1" ]]; then
	        printf -- '--ag-fold-dir %q --fasta %q' "${AG_FOLD_DIR}" "${FASTA}"
	      fi)

	  run_hound_data "${data_dir}"
	  if [[ "${USE_AG_FOLD}" == "1" ]]; then
	    train_data_dir="${data_dir}"
	  else
	    setup_folds "${sample_dir}" "${params_dir}/borzoi_linear.json" "${data_dir}"
	  fi

  for mode in ${MODES}; do
    params_file="${params_dir}/borzoi_${mode}.json"
    for trunk in "${TRUNKS[@]}"; do
      trunk_label="$(basename "${trunk}" .h5)"
      out_dir="${WORK_ROOT}/runs/${sample}/borzoi_${mode}/${trunk_label}"

      if [[ -f "${out_dir}/model_best.h5" && "${FORCE}" != "1" ]]; then
        echo "Skipping completed run: ${out_dir}"
        continue
      fi

      echo "Training ${sample} ${mode} ${trunk_label} -> ${out_dir}"
      transfer_cmd=(
        python src/baskerville/scripts/hound_transfer.py
        -o "${out_dir}"
        --trunk
        --restore "${trunk}"
      )
      if [[ "${MIXED_PRECISION}" == "1" ]]; then
        transfer_cmd+=(--mixed_precision)
      fi
      if [[ "${SKIP_TRAIN}" == "1" ]]; then
        transfer_cmd+=(--skip_train)
      fi
      transfer_cmd+=("${params_file}" "${train_data_dir}")
      "${transfer_cmd[@]}"
    done
  done
done
