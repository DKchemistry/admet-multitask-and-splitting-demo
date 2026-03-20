import os
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_method", required=True)
    ap.add_argument("--model_family", required=True)
    args = ap.parse_args()

    rows = []

    for iter_name in sorted(os.listdir(args.metrics_root)):
        if not iter_name.startswith("iter_"):
            continue

        cv_iter = int(iter_name.replace("iter_", ""))

        iter_dir = os.path.join(args.metrics_root, iter_name)

        for fold_name in sorted(os.listdir(iter_dir)):
            if not fold_name.startswith("fold_"):
                continue

            fold = int(fold_name.replace("fold_", ""))

            metrics_csv = os.path.join(iter_dir, fold_name, "metrics_all_targets.csv")
            if not os.path.isfile(metrics_csv):
                raise FileNotFoundError(f"Missing metrics file: {metrics_csv}")

            df = pd.read_csv(metrics_csv).copy()
            df["split_method"] = args.split_method
            df["model_family"] = args.model_family
            df["cv_iter"] = cv_iter
            df["fold"] = fold

            rows.append(df)

    if not rows:
        raise ValueError("No metrics files were found.")

    out_df = pd.concat(rows, axis=0, ignore_index=True)

    # We make target names shorter and more consistent for later plotting and stats
    target_map = {
        "HLM CLint": "HLM",
        "MLM CLint": "MLM",
    }
    out_df["target"] = out_df["target"].replace(target_map)

    # We build the comparison label we will likely use later in Tukey
    # Examples:
    #   HLM_RF_ECFP4_Butina
    #   MLM_RF_ECFP4_Butina
    out_df["comparison_label"] = (
        out_df["target"] + "_" + out_df["model_family"] + "_" + out_df["split_method"]
    )

    os.makedirs(args.out_dir, exist_ok=True)

    out_csv = os.path.join(args.out_dir, "rf_ecfp4_metrics_summary.csv")
    out_df.to_csv(out_csv, index=False)

    print(f"Wrote {out_csv} with {len(out_df)} rows")


if __name__ == "__main__":
    main()
