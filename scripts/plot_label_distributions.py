import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_numeric_values(csv_path, y_col):
    df = pd.read_csv(csv_path)
    values = pd.to_numeric(df[y_col], errors="coerce").dropna()
    values = values[values >= 0]
    return values


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--raw_train_csv", required=True)
    ap.add_argument("--processed_train_csv", required=True)
    ap.add_argument("--y_cols", required=True)
    ap.add_argument(
        "--out_png", default="figures/hlm_mlm_log10_distributions_train_only.png"
    )

    args = ap.parse_args()

    y_cols = [c.strip() for c in args.y_cols.split(",") if c.strip()]
    if len(y_cols) != 2:
        raise ValueError(
            "script expects exactly 2 label columns in --y_cols"
        )

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    raw_1 = load_numeric_values(args.raw_train_csv, y_cols[0])
    proc_1 = load_numeric_values(args.processed_train_csv, y_cols[0])

    raw_2 = load_numeric_values(args.raw_train_csv, y_cols[1])
    proc_2 = load_numeric_values(args.processed_train_csv, y_cols[1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(raw_1, bins=50)
    axes[0, 0].set_title(f"{y_cols[0]} raw train")
    axes[0, 0].set_xlabel("value")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(proc_1, bins=50)
    axes[0, 1].set_title(f"{y_cols[0]} log10(x+1) train")
    axes[0, 1].set_xlabel("value")
    axes[0, 1].set_ylabel("count")

    axes[1, 0].hist(raw_2, bins=50)
    axes[1, 0].set_title(f"{y_cols[1]} raw train")
    axes[1, 0].set_xlabel("value")
    axes[1, 0].set_ylabel("count")

    axes[1, 1].hist(proc_2, bins=50)
    axes[1, 1].set_title(f"{y_cols[1]} log10(x+1) train")
    axes[1, 1].set_xlabel("value")
    axes[1, 1].set_ylabel("count")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
