#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/oak/stanford/groups/akundaje/abuen/ag/baskerville/bash/submit_borzoi_atac_transfer_array.sbatch"
ROOT="${ROOT:-/oak/stanford/groups/akundaje/abuen/ag}"
TRUNK_ROOT="${TRUNK_ROOT:-${ROOT}/ag-data/borzoi/pretrain_trunks}"
TRUNK_GLOB="${TRUNK_GLOB:-trunk_r*.h5}"
SAMPLES="${SAMPLES:-K562 GM12878}"
MODES="${MODES:-lora}"

samples=(${SAMPLES})
modes=(${MODES})
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

array_max=$((${#samples[@]} * ${#modes[@]} * ${#trunks[@]} - 1))

export ROOT TRUNK_ROOT TRUNK_GLOB SAMPLES MODES

sbatch \
  --partition="${PARTITION:-akundaje}" \
  --gres="${GRES:-gpu:1}" \
  --constraint="${CONSTRAINT:-GPU_SKU:L40S|GPU_SKU:A100_SXM4}" \
  --array="0-${array_max}" \
  "${SCRIPT}"
