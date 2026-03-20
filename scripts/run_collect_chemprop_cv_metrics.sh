#!/usr/bin/env bash
set -euo pipefail

FOLD_ROOT_BASE="data/splits"
RESULTS_ROOT="results/hlm_mlm_cv_compare"

SMILES_COL="SMILES"
HLM_COL="HLM CLint"
MLM_COL="MLM CLint"

for SPLIT_NAME in scaffold butina random; do
  PREPARED_FOLD_ROOT="${FOLD_ROOT_BASE}/hlm_mlm_${SPLIT_NAME}_5x5cv_chemprop"

  case "${SPLIT_NAME}" in
    scaffold)
      SPLIT_LABEL="Scaffold"
      ;;
    butina)
      SPLIT_LABEL="Butina"
      ;;
    random)
      SPLIT_LABEL="Random"
      ;;
    *)
      echo "Unknown split name: ${SPLIT_NAME}"
      exit 1
      ;;
  esac

  echo "========================================"
  echo "Collecting metrics for split scheme: ${SPLIT_NAME}"
  echo "Prepared fold root: ${PREPARED_FOLD_ROOT}"
  echo "Results root: ${RESULTS_ROOT}/${SPLIT_NAME}"
  echo "========================================"

  echo
  echo "Single-task HLM"
  python scripts/collect_chemprop_cv_metrics.py \
    --prepared_fold_root "${PREPARED_FOLD_ROOT}" \
    --preds_root "${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_single/hlm" \
    --out_dir "${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_single/hlm/summary" \
    --split_method "${SPLIT_LABEL}" \
    --model_family "ChemProp_ST" \
    --smiles_col "${SMILES_COL}" \
    --target_cols "${HLM_COL}"

  echo
  echo "Single-task MLM"
  python scripts/collect_chemprop_cv_metrics.py \
    --prepared_fold_root "${PREPARED_FOLD_ROOT}" \
    --preds_root "${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_single/mlm" \
    --out_dir "${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_single/mlm/summary" \
    --split_method "${SPLIT_LABEL}" \
    --model_family "ChemProp_ST" \
    --smiles_col "${SMILES_COL}" \
    --target_cols "${MLM_COL}"

  echo
  echo "Multi-task HLM + MLM"
  python scripts/collect_chemprop_cv_metrics.py \
    --prepared_fold_root "${PREPARED_FOLD_ROOT}" \
    --preds_root "${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_multi/hlm_mlm" \
    --out_dir "${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_multi/hlm_mlm/summary" \
    --split_method "${SPLIT_LABEL}" \
    --model_family "ChemProp_MT" \
    --smiles_col "${SMILES_COL}" \
    --target_cols "${HLM_COL}" "${MLM_COL}"

  echo
  echo "Single-task HLM | foundation"
  python scripts/collect_chemprop_cv_metrics.py \
    --prepared_fold_root "${PREPARED_FOLD_ROOT}" \
    --preds_root "${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_single_foundation/hlm" \
    --out_dir "${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_single_foundation/hlm/summary" \
    --split_method "${SPLIT_LABEL}" \
    --model_family "ChemProp_ST_FM" \
    --smiles_col "${SMILES_COL}" \
    --target_cols "${HLM_COL}"

  echo
  echo "Single-task MLM | foundation"
  python scripts/collect_chemprop_cv_metrics.py \
    --prepared_fold_root "${PREPARED_FOLD_ROOT}" \
    --preds_root "${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_single_foundation/mlm" \
    --out_dir "${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_single_foundation/mlm/summary" \
    --split_method "${SPLIT_LABEL}" \
    --model_family "ChemProp_ST_FM" \
    --smiles_col "${SMILES_COL}" \
    --target_cols "${MLM_COL}"

  echo
  echo "Multi-task HLM + MLM | foundation"
  python scripts/collect_chemprop_cv_metrics.py \
    --prepared_fold_root "${PREPARED_FOLD_ROOT}" \
    --preds_root "${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_multi_foundation/hlm_mlm" \
    --out_dir "${RESULTS_ROOT}/${SPLIT_NAME}/chemprop_multi_foundation/hlm_mlm/summary" \
    --split_method "${SPLIT_LABEL}" \
    --model_family "ChemProp_MT_FM" \
    --smiles_col "${SMILES_COL}" \
    --target_cols "${HLM_COL}" "${MLM_COL}"
done

echo
echo "Done."