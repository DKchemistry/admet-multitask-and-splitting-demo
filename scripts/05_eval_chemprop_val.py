import os
import numpy as np
import pandas as pd

SMILES_COL = "SMILES"
PAPP_COL = "Caco-2 Permeability Papp A>B"
EFFLUX_COL = "Caco-2 Permeability Efflux"

VAL_TRUE = "data/processed/caco2_val_paired_temporal.csv"
SEEDS = [0, 1, 2, 3, 4]

SINGLE_PAPP_DIR = "results/chemprop_single/papp"
SINGLE_EFFLUX_DIR = "results/chemprop_single/efflux"
MULTI_DIR = "results/chemprop_multi"


def mae(y, yhat):
    return float(np.mean(np.abs(yhat - y)))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def r2(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def compute_metrics(y, yhat):
    return {"mae": mae(y, yhat), "rmse": rmse(y, yhat), "r2": r2(y, yhat)}


def merge_true_pred(true_df, pred_df, col):
    # Keep only SMILES + the column we care about from predictions
    pred_df = pred_df[[SMILES_COL, col]].copy()

    merged = true_df.merge(pred_df, on=SMILES_COL, how="inner", suffixes=("_true", "_pred"))

    y = merged[col + "_true"].values
    yhat = merged[col + "_pred"].values
    return merged, y, yhat


def main():
    true_val = pd.read_csv(VAL_TRUE)[[SMILES_COL, PAPP_COL, EFFLUX_COL]].copy()

    rows = []

    for seed in SEEDS:
        # --- single-task Papp
        path = os.path.join(SINGLE_PAPP_DIR, f"seed_{seed}", "model_0", "val_predictions.csv")
        pred = pd.read_csv(path)
        _, y, yhat = merge_true_pred(true_val, pred, PAPP_COL)
        rows.append({"seed": seed, "model": "single", "task": "papp", **compute_metrics(y, yhat)})

        # --- single-task Efflux
        path = os.path.join(SINGLE_EFFLUX_DIR, f"seed_{seed}", "model_0", "val_predictions.csv")
        pred = pd.read_csv(path)
        _, y, yhat = merge_true_pred(true_val, pred, EFFLUX_COL)
        rows.append({"seed": seed, "model": "single", "task": "efflux", **compute_metrics(y, yhat)})

        # --- multi-task (Papp + Efflux)
        path = os.path.join(MULTI_DIR, f"seed_{seed}", "model_0", "val_predictions.csv")
        pred = pd.read_csv(path)

        _, y, yhat = merge_true_pred(true_val, pred, PAPP_COL)
        rows.append({"seed": seed, "model": "multi", "task": "papp", **compute_metrics(y, yhat)})

        _, y, yhat = merge_true_pred(true_val, pred, EFFLUX_COL)
        rows.append({"seed": seed, "model": "multi", "task": "efflux", **compute_metrics(y, yhat)})

    df = pd.DataFrame(rows)

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    by_seed_path = os.path.join(out_dir, "chemprop_val_metrics_by_seed.csv")
    df = df.round({"mae": 4, "rmse": 4, "r2": 4})
    df.to_csv(by_seed_path, index=False)

    summary = df.groupby(["model", "task"], as_index=False)[["mae", "rmse", "r2"]].agg(["mean", "std"])
    summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.values]
    summary = summary.round(4)

    summary_path = os.path.join(out_dir, "chemprop_val_metrics_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("Wrote:")
    print(" -", by_seed_path)
    print(" -", summary_path)


if __name__ == "__main__":
    main()
