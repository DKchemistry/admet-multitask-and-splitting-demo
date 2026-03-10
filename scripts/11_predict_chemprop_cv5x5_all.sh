#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="MIG-75c6e677-9d93-5114-97a4-dab667418517"

SMILES_COL="SMILES"
N_REPEATS=5
N_FOLDS=5

predict_one () {
  local TEST_PATH="$1"
  local CKPT_PATH="$2"
  local OUT_PATH="$3"

  # Skip if checkpoint missing (training incomplete)
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "SKIP (missing ckpt): ${CKPT_PATH}"
    return 0
  fi

  # Skip if already predicted
  if [[ -f "${OUT_PATH}" ]]; then
    echo "SKIP (exists): ${OUT_PATH}"
    return 0
  fi

  echo "PREDICT -> ${OUT_PATH}"
  conda run -n chemprop chemprop predict \
    --test-path "${TEST_PATH}" \
    --smiles-columns "${SMILES_COL}" \
    --model-path "${CKPT_PATH}" \
    --preds-path "${OUT_PATH}"
}

for CV_ITER in $(seq 0 $((N_REPEATS-1))); do
  for FOLD in $(seq 0 $((N_FOLDS-1))); do
    TEST="data/cv5x5/iter_${CV_ITER}/fold_${FOLD}.csv"

    # ---- Multi-task ----
    CKPT_MULTI="results/cv5x5/chemprop_multi/iter_${CV_ITER}/fold_${FOLD}/model_0/best.pt"
    OUT_MULTI="results/cv5x5/chemprop_multi/iter_${CV_ITER}/fold_${FOLD}/predictions.csv"
    predict_one "${TEST}" "${CKPT_MULTI}" "${OUT_MULTI}"

    # ---- Single-task Papp ----
    CKPT_PAPP="results/cv5x5/chemprop_single/papp/iter_${CV_ITER}/fold_${FOLD}/model_0/best.pt"
    OUT_PAPP="results/cv5x5/chemprop_single/papp/iter_${CV_ITER}/fold_${FOLD}/predictions.csv"
    predict_one "${TEST}" "${CKPT_PAPP}" "${OUT_PAPP}"

    # ---- Single-task Efflux ----
    CKPT_EFF="results/cv5x5/chemprop_single/efflux/iter_${CV_ITER}/fold_${FOLD}/model_0/best.pt"
    OUT_EFF="results/cv5x5/chemprop_single/efflux/iter_${CV_ITER}/fold_${FOLD}/predictions.csv"
    predict_one "${TEST}" "${CKPT_EFF}" "${OUT_EFF}"
  done
done

echo "Done."