#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="MIG-75c6e677-9d93-5114-97a4-dab667418517"

SMILES_COL="SMILES"
HLM_COL="HLM CLint"
MLM_COL="MLM CLint"

SEED=0
N_REPEATS=5
N_FOLDS=5

for CV_ITER in $(seq 0 $((N_REPEATS-1))); do
  for FOLD in $(seq 0 $((N_FOLDS-1))); do
    DATA="data/cv5x5_hlm_mlm/iter_${CV_ITER}/fold_${FOLD}.csv"

    # ==============================
    # Single-task HLM
    # ==============================
    echo "=============================="
    echo "Single-task HLM | iter ${CV_ITER} fold ${FOLD} | seed ${SEED}"
    echo "=============================="

    OUT_HLM="results/cv5x5_hlm_mlm/chemprop_single/hlm/iter_${CV_ITER}/fold_${FOLD}"
    mkdir -p "${OUT_HLM}"

    conda run -n chemprop chemprop train \
      --data-path "${DATA}" \
      --splits-column split \
      --task-type regression \
      --smiles-columns "${SMILES_COL}" \
      --target-columns "${HLM_COL}" \
      --pytorch-seed "${SEED}" \
      --accelerator gpu \
      --devices 1 \
      --remove-checkpoints \
      --output-dir "${OUT_HLM}"

    # ==============================
    # Single-task MLM
    # ==============================
    echo "=============================="
    echo "Single-task MLM | iter ${CV_ITER} fold ${FOLD} | seed ${SEED}"
    echo "=============================="

    OUT_MLM="results/cv5x5_hlm_mlm/chemprop_single/mlm/iter_${CV_ITER}/fold_${FOLD}"
    mkdir -p "${OUT_MLM}"

    conda run -n chemprop chemprop train \
      --data-path "${DATA}" \
      --splits-column split \
      --task-type regression \
      --smiles-columns "${SMILES_COL}" \
      --target-columns "${MLM_COL}" \
      --pytorch-seed "${SEED}" \
      --accelerator gpu \
      --devices 1 \
      --remove-checkpoints \
      --output-dir "${OUT_MLM}"

    # ==============================
    # Multi-task (HLM + MLM)
    # ==============================
    echo "=============================="
    echo "Multi-task (HLM + MLM) | iter ${CV_ITER} fold ${FOLD} | seed ${SEED}"
    echo "=============================="

    OUT_MULTI="results/cv5x5_hlm_mlm/chemprop_multi/iter_${CV_ITER}/fold_${FOLD}"
    mkdir -p "${OUT_MULTI}"

    conda run -n chemprop chemprop train \
      --data-path "${DATA}" \
      --splits-column split \
      --task-type regression \
      --smiles-columns "${SMILES_COL}" \
      --target-columns "${HLM_COL}" "${MLM_COL}" \
      --pytorch-seed "${SEED}" \
      --accelerator gpu \
      --devices 1 \
      --remove-checkpoints \
      --output-dir "${OUT_MULTI}"

    # ==============================
    # Predictions (run after training)
    # ==============================
    echo "=============================="
    echo "Predict | iter ${CV_ITER} fold ${FOLD}"
    echo "=============================="

    TEST="${DATA}"

    CKPT_MULTI="${OUT_MULTI}/model_0/best.pt"
    OUT_MULTI_PRED="${OUT_MULTI}/predictions.csv"
    conda run -n chemprop chemprop predict \
      --test-path "${TEST}" \
      --smiles-columns "${SMILES_COL}" \
      --model-path "${CKPT_MULTI}" \
      --preds-path "${OUT_MULTI_PRED}"

    CKPT_HLM="${OUT_HLM}/model_0/best.pt"
    OUT_HLM_PRED="${OUT_HLM}/predictions.csv"
    conda run -n chemprop chemprop predict \
      --test-path "${TEST}" \
      --smiles-columns "${SMILES_COL}" \
      --model-path "${CKPT_HLM}" \
      --preds-path "${OUT_HLM_PRED}"

    CKPT_MLM="${OUT_MLM}/model_0/best.pt"
    OUT_MLM_PRED="${OUT_MLM}/predictions.csv"
    conda run -n chemprop chemprop predict \
      --test-path "${TEST}" \
      --smiles-columns "${SMILES_COL}" \
      --model-path "${CKPT_MLM}" \
      --preds-path "${OUT_MLM_PRED}"
  done
done

echo "Done."
