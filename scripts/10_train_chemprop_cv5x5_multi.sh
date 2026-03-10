#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="MIG-75c6e677-9d93-5114-97a4-dab667418517"

SMILES_COL="SMILES"
PAPP_COL="Caco-2 Permeability Papp A>B"
EFFLUX_COL="Caco-2 Permeability Efflux"

SEED=0

N_REPEATS=5
N_FOLDS=5

for CV_ITER in $(seq 0 $((N_REPEATS-1))); do
  for FOLD in $(seq 0 $((N_FOLDS-1))); do
    DATA="data/cv5x5/iter_${CV_ITER}/fold_${FOLD}.csv"

    echo "=============================="
    echo "Multi-task (Papp + Efflux) | iter ${CV_ITER} fold ${FOLD} | seed ${SEED}"
    echo "=============================="

    OUTDIR="results/cv5x5/chemprop_multi/iter_${CV_ITER}/fold_${FOLD}"
    mkdir -p "${OUTDIR}"

    conda run -n chemprop chemprop train \
      --data-path "${DATA}" \
      --splits-column split \
      --task-type regression \
      --smiles-columns "${SMILES_COL}" \
      --target-columns "${PAPP_COL}" "${EFFLUX_COL}" \
      --pytorch-seed "${SEED}" \
      --accelerator gpu \
      --devices 1 \
      --remove-checkpoints \
      --output-dir "${OUTDIR}"
  done
done