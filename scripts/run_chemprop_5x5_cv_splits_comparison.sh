#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="results/hlm_mlm_cv_compare"

SMILES_COL="SMILES"
HLM_COL="HLM CLint"
MLM_COL="MLM CLint"

SEED=0
N_REPEATS=5
N_FOLDS=5

run_st_hlm() {
  export CUDA_VISIBLE_DEVICES="MIG-b9001ecc-e23c-5123-902c-2f8401cf5448"

  for SPLIT_NAME in scaffold butina random; do
    FOLD_ROOT="data/splits/hlm_mlm_${SPLIT_NAME}_5x5cv_chemprop"

    for CV_ITER in $(seq 0 $((N_REPEATS - 1))); do
      for FOLD in $(seq 0 $((N_FOLDS - 1))); do
        TRAINVAL="${FOLD_ROOT}/iter_${CV_ITER}/fold_${FOLD}/trainval.csv"
        TEST="${FOLD_ROOT}/iter_${CV_ITER}/fold_${FOLD}/test.csv"
        OUT_DIR="${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_single/hlm/iter_${CV_ITER}/fold_${FOLD}"

        echo
        echo "========================================"
        echo "ChemProp ST HLM | ${SPLIT_NAME} | iter ${CV_ITER} fold ${FOLD}"
        echo "device  : ${CUDA_VISIBLE_DEVICES}"
        echo "trainval: ${TRAINVAL}"
        echo "test    : ${TEST}"
        echo "out     : ${OUT_DIR}"
        echo "========================================"

        [[ -f "${TRAINVAL}" ]] || { echo "Missing ${TRAINVAL}"; exit 1; }
        [[ -f "${TEST}" ]] || { echo "Missing ${TEST}"; exit 1; }

        mkdir -p "${OUT_DIR}"

        conda run -n chemprop chemprop train \
          --data-path "${TRAINVAL}" \
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
          --test-path "${TEST}" \
          --smiles-columns "${SMILES_COL}" \
          --model-path "${OUT_DIR}/model_0/best.pt" \
          --preds-path "${OUT_DIR}/predictions.csv"
      done
    done
  done
}

run_st_mlm() {
  export CUDA_VISIBLE_DEVICES="MIG-d411d40c-22f8-5184-9a88-3509c249fbbb"

  for SPLIT_NAME in scaffold butina random; do
    FOLD_ROOT="data/splits/hlm_mlm_${SPLIT_NAME}_5x5cv_chemprop"

    for CV_ITER in $(seq 0 $((N_REPEATS - 1))); do
      for FOLD in $(seq 0 $((N_FOLDS - 1))); do
        TRAINVAL="${FOLD_ROOT}/iter_${CV_ITER}/fold_${FOLD}/trainval.csv"
        TEST="${FOLD_ROOT}/iter_${CV_ITER}/fold_${FOLD}/test.csv"
        OUT_DIR="${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_single/mlm/iter_${CV_ITER}/fold_${FOLD}"

        echo
        echo "========================================"
        echo "ChemProp ST MLM | ${SPLIT_NAME} | iter ${CV_ITER} fold ${FOLD}"
        echo "device  : ${CUDA_VISIBLE_DEVICES}"
        echo "trainval: ${TRAINVAL}"
        echo "test    : ${TEST}"
        echo "out     : ${OUT_DIR}"
        echo "========================================"

        [[ -f "${TRAINVAL}" ]] || { echo "Missing ${TRAINVAL}"; exit 1; }
        [[ -f "${TEST}" ]] || { echo "Missing ${TEST}"; exit 1; }

        mkdir -p "${OUT_DIR}"

        conda run -n chemprop chemprop train \
          --data-path "${TRAINVAL}" \
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
          --test-path "${TEST}" \
          --smiles-columns "${SMILES_COL}" \
          --model-path "${OUT_DIR}/model_0/best.pt" \
          --preds-path "${OUT_DIR}/predictions.csv"
      done
    done
  done
}

run_mt() {
  export CUDA_VISIBLE_DEVICES="MIG-d2513014-4073-5b89-b98d-39651c1427c8"

  for SPLIT_NAME in scaffold butina random; do
    FOLD_ROOT="data/splits/hlm_mlm_${SPLIT_NAME}_5x5cv_chemprop"

    for CV_ITER in $(seq 0 $((N_REPEATS - 1))); do
      for FOLD in $(seq 0 $((N_FOLDS - 1))); do
        TRAINVAL="${FOLD_ROOT}/iter_${CV_ITER}/fold_${FOLD}/trainval.csv"
        TEST="${FOLD_ROOT}/iter_${CV_ITER}/fold_${FOLD}/test.csv"
        OUT_DIR="${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_multi/hlm_mlm/iter_${CV_ITER}/fold_${FOLD}"

        echo
        echo "========================================"
        echo "ChemProp MT HLM+MLM | ${SPLIT_NAME} | iter ${CV_ITER} fold ${FOLD}"
        echo "device  : ${CUDA_VISIBLE_DEVICES}"
        echo "trainval: ${TRAINVAL}"
        echo "test    : ${TEST}"
        echo "out     : ${OUT_DIR}"
        echo "========================================"

        [[ -f "${TRAINVAL}" ]] || { echo "Missing ${TRAINVAL}"; exit 1; }
        [[ -f "${TEST}" ]] || { echo "Missing ${TEST}"; exit 1; }

        mkdir -p "${OUT_DIR}"

        conda run -n chemprop chemprop train \
          --data-path "${TRAINVAL}" \
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
          --test-path "${TEST}" \
          --smiles-columns "${SMILES_COL}" \
          --model-path "${OUT_DIR}/model_0/best.pt" \
          --preds-path "${OUT_DIR}/predictions.csv"
      done
    done
  done
}

run_st_fm_hlm() {
  export CUDA_VISIBLE_DEVICES="MIG-a1a9682b-034e-5089-b2e4-84399f87b472"

  for SPLIT_NAME in scaffold butina random; do
    FOLD_ROOT="data/splits/hlm_mlm_${SPLIT_NAME}_5x5cv_chemprop"

    for CV_ITER in $(seq 0 $((N_REPEATS - 1))); do
      for FOLD in $(seq 0 $((N_FOLDS - 1))); do
        TRAINVAL="${FOLD_ROOT}/iter_${CV_ITER}/fold_${FOLD}/trainval.csv"
        TEST="${FOLD_ROOT}/iter_${CV_ITER}/fold_${FOLD}/test.csv"
        OUT_DIR="${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_single_foundation/hlm/iter_${CV_ITER}/fold_${FOLD}"

        echo
        echo "========================================"
        echo "ChemProp ST + FM HLM | ${SPLIT_NAME} | iter ${CV_ITER} fold ${FOLD}"
        echo "device  : ${CUDA_VISIBLE_DEVICES}"
        echo "trainval: ${TRAINVAL}"
        echo "test    : ${TEST}"
        echo "out     : ${OUT_DIR}"
        echo "========================================"

        [[ -f "${TRAINVAL}" ]] || { echo "Missing ${TRAINVAL}"; exit 1; }
        [[ -f "${TEST}" ]] || { echo "Missing ${TEST}"; exit 1; }

        mkdir -p "${OUT_DIR}"

        conda run -n chemprop chemprop train \
          --data-path "${TRAINVAL}" \
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
          --test-path "${TEST}" \
          --smiles-columns "${SMILES_COL}" \
          --model-path "${OUT_DIR}/model_0/best.pt" \
          --preds-path "${OUT_DIR}/predictions.csv"
      done
    done
  done
}

run_st_fm_mlm() {
  export CUDA_VISIBLE_DEVICES="MIG-d6d882a2-eede-5cfc-8b94-a6f8c9097c31"

  for SPLIT_NAME in scaffold butina random; do
    FOLD_ROOT="data/splits/hlm_mlm_${SPLIT_NAME}_5x5cv_chemprop"

    for CV_ITER in $(seq 0 $((N_REPEATS - 1))); do
      for FOLD in $(seq 0 $((N_FOLDS - 1))); do
        TRAINVAL="${FOLD_ROOT}/iter_${CV_ITER}/fold_${FOLD}/trainval.csv"
        TEST="${FOLD_ROOT}/iter_${CV_ITER}/fold_${FOLD}/test.csv"
        OUT_DIR="${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_single_foundation/mlm/iter_${CV_ITER}/fold_${FOLD}"

        echo
        echo "========================================"
        echo "ChemProp ST + FM MLM | ${SPLIT_NAME} | iter ${CV_ITER} fold ${FOLD}"
        echo "device  : ${CUDA_VISIBLE_DEVICES}"
        echo "trainval: ${TRAINVAL}"
        echo "test    : ${TEST}"
        echo "out     : ${OUT_DIR}"
        echo "========================================"

        [[ -f "${TRAINVAL}" ]] || { echo "Missing ${TRAINVAL}"; exit 1; }
        [[ -f "${TEST}" ]] || { echo "Missing ${TEST}"; exit 1; }

        mkdir -p "${OUT_DIR}"

        conda run -n chemprop chemprop train \
          --data-path "${TRAINVAL}" \
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
          --test-path "${TEST}" \
          --smiles-columns "${SMILES_COL}" \
          --model-path "${OUT_DIR}/model_0/best.pt" \
          --preds-path "${OUT_DIR}/predictions.csv"
      done
    done
  done
}

run_mt_fm() {
  export CUDA_VISIBLE_DEVICES="MIG-3b8da2bd-5073-580e-8330-c7167817bcd7"

  for SPLIT_NAME in scaffold butina random; do
    FOLD_ROOT="data/splits/hlm_mlm_${SPLIT_NAME}_5x5cv_chemprop"

    for CV_ITER in $(seq 0 $((N_REPEATS - 1))); do
      for FOLD in $(seq 0 $((N_FOLDS - 1))); do
        TRAINVAL="${FOLD_ROOT}/iter_${CV_ITER}/fold_${FOLD}/trainval.csv"
        TEST="${FOLD_ROOT}/iter_${CV_ITER}/fold_${FOLD}/test.csv"
        OUT_DIR="${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_multi_foundation/hlm_mlm/iter_${CV_ITER}/fold_${FOLD}"

        echo
        echo "========================================"
        echo "ChemProp MT + FM HLM+MLM | ${SPLIT_NAME} | iter ${CV_ITER} fold ${FOLD}"
        echo "device  : ${CUDA_VISIBLE_DEVICES}"
        echo "trainval: ${TRAINVAL}"
        echo "test    : ${TEST}"
        echo "out     : ${OUT_DIR}"
        echo "========================================"

        [[ -f "${TRAINVAL}" ]] || { echo "Missing ${TRAINVAL}"; exit 1; }
        [[ -f "${TEST}" ]] || { echo "Missing ${TEST}"; exit 1; }

        mkdir -p "${OUT_DIR}"

        conda run -n chemprop chemprop train \
          --data-path "${TRAINVAL}" \
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
          --test-path "${TEST}" \
          --smiles-columns "${SMILES_COL}" \
          --model-path "${OUT_DIR}/model_0/best.pt" \
          --preds-path "${OUT_DIR}/predictions.csv"
      done
    done
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