import argparse
from pathlib import Path

import pandas as pd

SMILES_COL = "SMILES"


def load_raw(path):
    df = pd.read_csv(path)
    return df


def clean_and_select(df, name, y_cols):
    print("\n===", name, "===")
    print("start rows:", len(df))

    cols = [SMILES_COL] + y_cols

    # Keep only the columns we care about 
    df = df[cols].copy()

    # Strip SMILES whitespace and treat empty string as missing
    df[SMILES_COL] = df[SMILES_COL].astype(str).str.strip()
    df.loc[df[SMILES_COL] == "", SMILES_COL] = pd.NA

    # Coerce labels to numeric. Any weird strings become NaN.
    for c in y_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing any required fields
    before = len(df)
    df = df.dropna(subset=cols).copy()
    print("dropped (missing/unparseable in required cols):", before - len(df))
    print("rows after dropna:", len(df))

    # Deduplicate by SMILES, keeping the first occurrence (assumed earliest/temporal)
    before = len(df)
    df = df.drop_duplicates(subset=[SMILES_COL], keep="first").copy()
    print("dropped duplicates by SMILES:", before - len(df))
    print("rows after dedupe:", len(df))

    # Final quick sanity
    print("final missing counts (should be 0):")
    print(df[cols].isna().sum().to_string())

    return df


def temporal_split(df, val_frac):
    n = len(df)
    n_val = int(round(n * val_frac))

    if n_val <= 0 or n_val >= n:
        raise ValueError(
            "val_frac results in empty train or empty val; choose a val_frac like 0.1-0.3"
        )

    # Temporal assumption: earlier rows -> train, later rows -> val
    train_df = df.iloc[: n - n_val].copy()
    val_df = df.iloc[n - n_val :].copy()

    return train_df, val_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/raw/caco2_train.csv")
    ap.add_argument("--test_csv", default="data/raw/caco2_test.csv")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--out_prefix", default="caco2")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument(
        "--y_cols", default="Caco-2 Permeability Papp A>B,Caco-2 Permeability Efflux"
    )
    args = ap.parse_args()

    y_cols = [c.strip() for c in args.y_cols.split(",") if c.strip()]
    if not y_cols:
        raise ValueError("--y_cols must contain at least one column name")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_train = load_raw(args.train_csv)
    raw_test = load_raw(args.test_csv)

    train_clean = clean_and_select(raw_train, "HF TRAIN raw -> cleaned", y_cols)
    test_clean = clean_and_select(raw_test, "HF TEST raw -> cleaned", y_cols)

    train_split, val_split = temporal_split(train_clean, args.val_frac)

    print("\n=== Temporal split summary (based on row order) ===")
    print("train rows:", len(train_split))
    print("val rows:  ", len(val_split))
    print(
        "val is last",
        int(round(len(train_clean) * args.val_frac)),
        "rows of cleaned train",
    )

    train_out = out_dir / f"{args.out_prefix}_train_paired_temporal.csv"
    val_out   = out_dir / f"{args.out_prefix}_val_paired_temporal.csv"
    test_out  = out_dir / f"{args.out_prefix}_test_paired.csv"

    train_split.to_csv(train_out, index=False)
    val_split.to_csv(val_out, index=False)
    test_clean.to_csv(test_out, index=False)

    print("\nWrote:")
    print(" ", train_out, "shape=", train_split.shape)
    print(" ", val_out, "shape=", val_split.shape)
    print(" ", test_out, "shape=", test_clean.shape)


if __name__ == "__main__":
    main()
