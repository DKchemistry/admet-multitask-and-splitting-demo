import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SMILES_COL = "SMILES"
PAPP_COL = "Caco-2 Permeability Papp A>B"
EFFLUX_COL = "Caco-2 Permeability Efflux"


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def load_val_truth(fold_csv_path):
    """Load validation rows (ground truth) from the fold CSV."""
    df = pd.read_csv(fold_csv_path)
    if "row_id" not in df.columns or "split" not in df.columns:
        raise ValueError(f"Expected row_id and split columns in {fold_csv_path}")
    val = df[df["split"] == "val"][["row_id", PAPP_COL, EFFLUX_COL]].copy()
    return val


def load_val_preds(pred_csv_path):
    """Load validation rows (predictions) from the prediction CSV."""
    df = pd.read_csv(pred_csv_path)
    if "row_id" not in df.columns or "split" not in df.columns:
        raise ValueError(f"Expected row_id and split columns in {pred_csv_path}")
    val = df[df["split"] == "val"].copy()
    return val


def eval_single_task(truth_val, preds_val, task_col):
    """
    Evaluate one task by joining on row_id.
    truth_val has true columns; preds_val has predicted column named task_col.
    """
    need_cols = {"row_id", task_col}
    if not need_cols.issubset(preds_val.columns):
        raise ValueError(
            f"Prediction file missing required columns {need_cols}. "
            f"Has: {list(preds_val.columns)}"
        )

    merged = truth_val.merge(
        preds_val[["row_id", task_col]],
        on="row_id",
        suffixes=("_true", "_pred"),
        how="inner",
    )

    y_true = merged[f"{task_col}_true"].values
    y_pred = merged[f"{task_col}_pred"].values
    m = compute_metrics(y_true, y_pred)
    m["n_val"] = int(len(merged))
    return m


def main():
    n_repeats = 5
    n_folds = 5

    # Inputs
    fold_data_root = "data/cv5x5"

    # Predictions roots
    multi_root = "results/cv5x5/chemprop_multi"
    single_root = "results/cv5x5/chemprop_single"

    # Output
    out_path = "results/cv5x5/chemprop_metrics_long.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []

    for cv_iter in range(n_repeats):
        for fold in range(n_folds):
            fold_csv = os.path.join(
                fold_data_root, f"iter_{cv_iter}", f"fold_{fold}.csv"
            )
            truth_val = load_val_truth(fold_csv)

            # --- Multi-task ---
            pred_multi = os.path.join(
                multi_root, f"iter_{cv_iter}", f"fold_{fold}", "predictions.csv"
            )
            preds_val_multi = load_val_preds(pred_multi)

            m_papp = eval_single_task(truth_val, preds_val_multi, PAPP_COL)
            rows.append(
                {
                    "model_family": "chemprop",
                    "variant": "multi",
                    "task": "papp",
                    "cv_iter": cv_iter,
                    "fold": fold,
                    **m_papp,
                }
            )

            m_eff = eval_single_task(truth_val, preds_val_multi, EFFLUX_COL)
            rows.append(
                {
                    "model_family": "chemprop",
                    "variant": "multi",
                    "task": "efflux",
                    "cv_iter": cv_iter,
                    "fold": fold,
                    **m_eff,
                }
            )

            # --- Single-task Papp ---
            pred_papp = os.path.join(
                single_root,
                "papp",
                f"iter_{cv_iter}",
                f"fold_{fold}",
                "predictions.csv",
            )
            preds_val_papp = load_val_preds(pred_papp)
            m = eval_single_task(truth_val, preds_val_papp, PAPP_COL)
            rows.append(
                {
                    "model_family": "chemprop",
                    "variant": "single_papp",
                    "task": "papp",
                    "cv_iter": cv_iter,
                    "fold": fold,
                    **m,
                }
            )

            # --- Single-task Efflux ---
            pred_eff = os.path.join(
                single_root,
                "efflux",
                f"iter_{cv_iter}",
                f"fold_{fold}",
                "predictions.csv",
            )
            preds_val_eff = load_val_preds(pred_eff)
            m = eval_single_task(truth_val, preds_val_eff, EFFLUX_COL)
            rows.append(
                {
                    "model_family": "chemprop",
                    "variant": "single_efflux",
                    "task": "efflux",
                    "cv_iter": cv_iter,
                    "fold": fold,
                    **m,
                }
            )

    metrics_df = pd.DataFrame(rows)

    # Optional: round for readability
    metrics_df = metrics_df.round({"mae": 4, "rmse": 4, "r2": 6})

    metrics_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    print(
        f"Rows: {len(metrics_df)} (expected {n_repeats*n_folds*4} = {n_repeats*n_folds*4})"
    )


if __name__ == "__main__":
    main()
