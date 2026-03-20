import os
import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau


def compute_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    tau, tau_pvalue = kendalltau(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "kendall_tau": tau,
        "kendall_tau_pvalue": tau_pvalue,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared_fold_root", required=True)
    ap.add_argument("--preds_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_method", required=True)
    ap.add_argument("--model_family", required=True)
    ap.add_argument("--smiles_col", default="SMILES")
    ap.add_argument("--target_cols", nargs="+", required=True)
    args = ap.parse_args()

    rows = []

    for iter_name in sorted(os.listdir(args.preds_root)):
        if not iter_name.startswith("iter_"):
            continue

        cv_iter = int(iter_name.replace("iter_", ""))
        iter_dir = os.path.join(args.preds_root, iter_name)

        for fold_name in sorted(os.listdir(iter_dir)):
            if not fold_name.startswith("fold_"):
                continue

            fold = int(fold_name.replace("fold_", ""))

            truth_csv = os.path.join(
                args.prepared_fold_root, iter_name, fold_name, "test.csv"
            )
            pred_csv = os.path.join(iter_dir, fold_name, "predictions.csv")

            if not os.path.isfile(truth_csv):
                raise FileNotFoundError(f"Missing truth file: {truth_csv}")

            if not os.path.isfile(pred_csv):
                raise FileNotFoundError(f"Missing predictions file: {pred_csv}")

            truth_df = pd.read_csv(truth_csv).copy()
            pred_df = pd.read_csv(pred_csv).copy()

            # test.csv already is the held-out evaluation set
            truth_keep = ["row_id", args.smiles_col] + args.target_cols
            pred_keep = ["row_id", args.smiles_col] + args.target_cols

            truth_df = truth_df[truth_keep].copy()
            pred_df = pred_df[pred_keep].copy()

            merged = truth_df.merge(
                pred_df,
                on=["row_id", args.smiles_col],
                suffixes=("_true", "_pred"),
            )

            if len(merged) == 0:
                raise ValueError(
                    f"No rows found after merge for iter {cv_iter} fold {fold}"
                )

            if len(merged) != len(truth_df):
                raise ValueError(
                    f"Merged row count does not match truth row count for "
                    f"iter {cv_iter} fold {fold}: "
                    f"merged={len(merged)} truth={len(truth_df)}"
                )

            if len(merged) != len(pred_df):
                raise ValueError(
                    f"Merged row count does not match prediction row count for "
                    f"iter {cv_iter} fold {fold}: "
                    f"merged={len(merged)} pred={len(pred_df)}"
                )

            for target_col in args.target_cols:
                true_col = f"{target_col}_true"
                pred_col = f"{target_col}_pred"

                score_df = merged[[true_col, pred_col]].dropna().copy()

                if len(score_df) == 0:
                    raise ValueError(
                        f"No non-missing rows for iter {cv_iter} fold {fold} "
                        f"target {target_col}"
                    )

                metrics = compute_metrics(
                    score_df[true_col].tolist(),
                    score_df[pred_col].tolist(),
                )

                rows.append(
                    {
                        "target": target_col,
                        "n_test": len(score_df),
                        "split_method": args.split_method,
                        "model_family": args.model_family,
                        "cv_iter": cv_iter,
                        "fold": fold,
                        **metrics,
                    }
                )

    if not rows:
        raise ValueError("No metrics were collected.")

    out_df = pd.DataFrame(rows)

    target_map = {
        "HLM CLint": "HLM",
        "MLM CLint": "MLM",
    }
    out_df["target"] = out_df["target"].replace(target_map)

    out_df["comparison_label"] = (
        out_df["target"] + "_" + out_df["model_family"] + "_" + out_df["split_method"]
    )

    os.makedirs(args.out_dir, exist_ok=True)

    out_csv = os.path.join(args.out_dir, "chemprop_metrics_summary.csv")
    out_df.to_csv(out_csv, index=False)

    print(f"Wrote {out_csv} with {len(out_df)} rows")


if __name__ == "__main__":
    main()
