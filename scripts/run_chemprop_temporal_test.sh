#!/usr/bin/env bash
set -euo pipefail

TRAIN_DATA="data/processed/hlm_mlm_paired_train_temporal_80_20.csv"
TEST_DATA="data/raw/hlm_mlm_test.csv"
RESULTS_ROOT="results/hlm_mlm_true_test"

SMILES_COL="SMILES"
HLM_COL="HLM CLint"
MLM_COL="MLM CLint"

SEED_START=0
SEED_END=24

run_st_hlm() {
  export CUDA_VISIBLE_DEVICES="MIG-d2513014-4073-5b89-b98d-39651c1427c8"

  for SEED in $(seq "${SEED_START}" "${SEED_END}"); do
    OUT_DIR="${RESULTS_ROOT}/chemprop_single/hlm/seed_${SEED}"

    echo
    echo "========================================"
    echo "ChemProp ST HLM | seed ${SEED}"
    echo "device: ${CUDA_VISIBLE_DEVICES}"
    echo "out   : ${OUT_DIR}"
    echo "========================================"

    mkdir -p "${OUT_DIR}"

    conda run -n chemprop chemprop train \
      --data-path "${TRAIN_DATA}" \
      --splits-column split \
      --task-type regression \
      --smiles-columns "${SMILES_COL}" \
      --target-columns "${HLM_COL}" \
      --pytorch-seed "${SEED}" \
      --accelerator gpu \
      --devices 1 \
      --remove-checkpoints \
      --output-dir "${OUT_DIR}"

    conda run -n chemprop chemprop predict \
      --test-path "${TEST_DATA}" \
      --smiles-columns "${SMILES_COL}" \
      --model-path "${OUT_DIR}/model_0/best.pt" \
      --preds-path "${OUT_DIR}/predictions.csv"
  done
}

run_st_mlm() {
  export CUDA_VISIBLE_DEVICES="MIG-a1a9682b-034e-5089-b2e4-84399f87b472"

  for SEED in $(seq "${SEED_START}" "${SEED_END}"); do
    OUT_DIR="${RESULTS_ROOT}/chemprop_single/mlm/seed_${SEED}"

    echo
    echo "========================================"
    echo "ChemProp ST MLM | seed ${SEED}"
    echo "device: ${CUDA_VISIBLE_DEVICES}"
    echo "out   : ${OUT_DIR}"
    echo "========================================"

    mkdir -p "${OUT_DIR}"

    conda run -n chemprop chemprop train \
      --data-path "${TRAIN_DATA}" \
      --splits-column split \
      --task-type regression \
      --smiles-columns "${SMILES_COL}" \
      --target-columns "${MLM_COL}" \
      --pytorch-seed "${SEED}" \
      --accelerator gpu \
      --devices 1 \
      --remove-checkpoints \
      --output-dir "${OUT_DIR}"

    conda run -n chemprop chemprop predict \
      --test-path "${TEST_DATA}" \
      --smiles-columns "${SMILES_COL}" \
      --model-path "${OUT_DIR}/model_0/best.pt" \
      --preds-path "${OUT_DIR}/predictions.csv"
  done
}

run_mt() {
  export CUDA_VISIBLE_DEVICES="MIG-3b8da2bd-5073-580e-8330-c7167817bcd7"

  for SEED in $(seq "${SEED_START}" "${SEED_END}"); do
    OUT_DIR="${RESULTS_ROOT}/chemprop_multi/hlm_mlm/seed_${SEED}"

    echo
    echo "========================================"
    echo "ChemProp MT HLM+MLM | seed ${SEED}"
    echo "device: ${CUDA_VISIBLE_DEVICES}"
    echo "out   : ${OUT_DIR}"
    echo "========================================"

    mkdir -p "${OUT_DIR}"

    conda run -n chemprop chemprop train \
      --data-path "${TRAIN_DATA}" \
      --splits-column split \
      --task-type regression \
      --smiles-columns "${SMILES_COL}" \
      --target-columns "${HLM_COL}" "${MLM_COL}" \
      --pytorch-seed "${SEED}" \
      --accelerator gpu \
      --devices 1 \
      --remove-checkpoints \
      --output-dir "${OUT_DIR}"

    conda run -n chemprop chemprop predict \
      --test-path "${TEST_DATA}" \
      --smiles-columns "${SMILES_COL}" \
      --model-path "${OUT_DIR}/model_0/best.pt" \
      --preds-path "${OUT_DIR}/predictions.csv"
  done
}

run_st_fm_hlm() {
  export CUDA_VISIBLE_DEVICES="MIG-b9001ecc-e23c-5123-902c-2f8401cf5448"

  for SEED in $(seq "${SEED_START}" "${SEED_END}"); do
    OUT_DIR="${RESULTS_ROOT}/chemprop_single_foundation/hlm/seed_${SEED}"

    echo
    echo "========================================"
    echo "ChemProp ST + FM HLM | seed ${SEED}"
    echo "device: ${CUDA_VISIBLE_DEVICES}"
    echo "out   : ${OUT_DIR}"
    echo "========================================"

    mkdir -p "${OUT_DIR}"

    conda run -n chemprop chemprop train \
      --data-path "${TRAIN_DATA}" \
      --splits-column split \
      --task-type regression \
      --smiles-columns "${SMILES_COL}" \
      --target-columns "${HLM_COL}" \
      --pytorch-seed "${SEED}" \
      --accelerator gpu \
      --devices 1 \
      --remove-checkpoints \
      --from-foundation CheMeleon \
      --output-dir "${OUT_DIR}"

    conda run -n chemprop chemprop predict \
      --test-path "${TEST_DATA}" \
      --smiles-columns "${SMILES_COL}" \
      --model-path "${OUT_DIR}/model_0/best.pt" \
      --preds-path "${OUT_DIR}/predictions.csv"
  done
}

run_st_fm_mlm() {
  export CUDA_VISIBLE_DEVICES="MIG-d411d40c-22f8-5184-9a88-3509c249fbbb"

  for SEED in $(seq "${SEED_START}" "${SEED_END}"); do
    OUT_DIR="${RESULTS_ROOT}/chemprop_single_foundation/mlm/seed_${SEED}"

    echo
    echo "========================================"
    echo "ChemProp ST + FM MLM | seed ${SEED}"
    echo "device: ${CUDA_VISIBLE_DEVICES}"
    echo "out   : ${OUT_DIR}"
    echo "========================================"

    mkdir -p "${OUT_DIR}"

    conda run -n chemprop chemprop train \
      --data-path "${TRAIN_DATA}" \
      --splits-column split \
      --task-type regression \
      --smiles-columns "${SMILES_COL}" \
      --target-columns "${MLM_COL}" \
      --pytorch-seed "${SEED}" \
      --accelerator gpu \
      --devices 1 \
      --remove-checkpoints \
      --from-foundation CheMeleon \
      --output-dir "${OUT_DIR}"

    conda run -n chemprop chemprop predict \
      --test-path "${TEST_DATA}" \
      --smiles-columns "${SMILES_COL}" \
      --model-path "${OUT_DIR}/model_0/best.pt" \
      --preds-path "${OUT_DIR}/predictions.csv"
  done
}

run_mt_fm() {
  export CUDA_VISIBLE_DEVICES="MIG-435ac987-77ee-564b-9844-8d69355885d2"

  for SEED in $(seq "${SEED_START}" "${SEED_END}"); do
    OUT_DIR="${RESULTS_ROOT}/chemprop_multi_foundation/hlm_mlm/seed_${SEED}"

    echo
    echo "========================================"
    echo "ChemProp MT + FM HLM+MLM | seed ${SEED}"
    echo "device: ${CUDA_VISIBLE_DEVICES}"
    echo "out   : ${OUT_DIR}"
    echo "========================================"

    mkdir -p "${OUT_DIR}"

    conda run -n chemprop chemprop train \
      --data-path "${TRAIN_DATA}" \
      --splits-column split \
      --task-type regression \
      --smiles-columns "${SMILES_COL}" \
      --target-columns "${HLM_COL}" "${MLM_COL}" \
      --pytorch-seed "${SEED}" \
      --accelerator gpu \
      --devices 1 \
      --remove-checkpoints \
      --from-foundation CheMeleon \
      --output-dir "${OUT_DIR}"

    conda run -n chemprop chemprop predict \
      --test-path "${TEST_DATA}" \
      --smiles-columns "${SMILES_COL}" \
      --model-path "${OUT_DIR}/model_0/best.pt" \
      --preds-path "${OUT_DIR}/predictions.csv"
  done
}

run_st_hlm &
PID_ST_HLM=$!

run_st_mlm &
PID_ST_MLM=$!

run_mt &
PID_MT=$!

run_st_fm_hlm &
PID_ST_FM_HLM=$!

run_st_fm_mlm &
PID_ST_FM_MLM=$!

run_mt_fm &
PID_MT_FM=$!

wait "${PID_ST_HLM}"
wait "${PID_ST_MLM}"
wait "${PID_MT}"
wait "${PID_ST_FM_HLM}"
wait "${PID_ST_FM_MLM}"
wait "${PID_MT_FM}"

echo
echo "Done."