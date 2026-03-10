#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="MIG-75c6e677-9d93-5114-97a4-dab667418517"

TRAIN="data/processed/caco2_train_paired_temporal.csv"
VAL="data/processed/caco2_val_paired_temporal.csv"
TEST="$VAL"   # intentional: avoid peeking at real test during dev

SMILES_COL="SMILES"
PAPP_COL="Caco-2 Permeability Papp A>B"
EFFLUX_COL="Caco-2 Permeability Efflux"

SEEDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do
  echo "=============================="
  echo "Single-task Papp | seed $SEED"
  echo "=============================="

  OUTDIR="results/chemprop_single/papp/seed_${SEED}"
  mkdir -p "$OUTDIR"

  chemprop train \
    --data-path "$TRAIN" "$VAL" "$TEST" \
    --task-type regression \
    --smiles-columns "$SMILES_COL" \
    --target-columns "$PAPP_COL" \
    --pytorch-seed "$SEED" \
    --accelerator gpu \
    --devices 1 \
    --remove-checkpoints \
    --output-dir "$OUTDIR"

  echo "=============================="
  echo "Single-task Efflux | seed $SEED"
  echo "=============================="

  OUTDIR="results/chemprop_single/efflux/seed_${SEED}"
  mkdir -p "$OUTDIR"

  chemprop train \
    --data-path "$TRAIN" "$VAL" "$TEST" \
    --task-type regression \
    --smiles-columns "$SMILES_COL" \
    --target-columns "$EFFLUX_COL" \
    --pytorch-seed "$SEED" \
    --accelerator gpu \
    --devices 1 \
    --remove-checkpoints \
    --output-dir "$OUTDIR"
done
