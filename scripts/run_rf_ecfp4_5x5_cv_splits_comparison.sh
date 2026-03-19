#!/usr/bin/env bash
set -euo pipefail

python scripts/run_rf_ecfp4_cv.py \
  --fold_root data/splits/hlm_mlm_scaffold_5x5cv/folds \
  --out_root results/hlm_mlm_cv_compare/scaffold/rf_ecfp4 \
  --smiles_col SMILES \
  --target_cols "HLM CLint" "MLM CLint" \
  --n_folds 5 \
  --n_repeats 5

python scripts/run_rf_ecfp4_cv.py \
  --fold_root data/splits/hlm_mlm_butina_5x5cv/folds \
  --out_root results/hlm_mlm_cv_compare/butina/rf_ecfp4 \
  --smiles_col SMILES \
  --target_cols "HLM CLint" "MLM CLint" \
  --n_folds 5 \
  --n_repeats 5

python scripts/run_rf_ecfp4_cv.py \
  --fold_root data/splits/hlm_mlm_random_5x5cv/folds \
  --out_root results/hlm_mlm_cv_compare/random/rf_ecfp4 \
  --smiles_col SMILES \
  --target_cols "HLM CLint" "MLM CLint" \
  --n_folds 5 \
  --n_repeats 5