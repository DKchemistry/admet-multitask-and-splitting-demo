import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SMILES_COL = "SMILES"


def clean_smiles_column(df):
    df = df.copy()

    # Remove leading/trailing whitespace
    df[SMILES_COL] = df[SMILES_COL].astype(str).str.strip()

    # Empty strings should count as missing
    df.loc[df[SMILES_COL] == "", SMILES_COL] = pd.NA

    return df


def coerce_label_columns(df, y_cols):
    df = df.copy()

    # Convert labels to numeric. Bad strings become NaN.
    for c in y_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def filter_paired_rows(df, y_cols):
    df = df.copy()

    before = len(df)

    # Keep only rows with SMILES present and all requested labels present
    df = df.dropna(subset=[SMILES_COL] + y_cols).copy()

    print("dropped rows during paired filtering:", before - len(df))
    print("rows after paired filtering:", len(df))

    return df


def log10_transform_labels(df, y_cols):
    df = df.copy()

    before = len(df)

    # log10(x + 1) keeps zeros, but negative values are invalid
    keep_mask = (df[y_cols] >= 0).all(axis=1)
    df = df[keep_mask].copy()

    print("dropped rows with negative labels for log10(x+1):", before - len(df))
    print("rows after log10(x+1) filter:", len(df))

    for c in y_cols:
        df[c] = np.log10(df[c] + 1)

    return df


def deduplicate_by_smiles(df):
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset=[SMILES_COL], keep="first").copy()

    print("dropped duplicates by SMILES:", before - len(df))
    print("rows after dedupe:", len(df))

    return df


def process_split(df, split_name, y_cols, log10):
    print("\n==============================")
    print(split_name)
    print("==============================")
    print("start rows:", len(df))

    cols = [SMILES_COL] + y_cols
    df = df[cols].copy()

    df = clean_smiles_column(df)
    df = coerce_label_columns(df, y_cols)

    print("\nMissing labels after numeric coercion:")
    print(df[y_cols].isna().sum().to_string())

    df = filter_paired_rows(df, y_cols)

    if log10:
        df = log10_transform_labels(df, y_cols)

    df = deduplicate_by_smiles(df)

    print("\nFinal missing counts:")
    print(df[cols].isna().sum().to_string())

    return df


def make_output_prefix(out_prefix, log10):
    parts = [out_prefix, "paired"]

    if log10:
        parts.append("log10")

    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--y_cols", required=True)
    ap.add_argument("--log10", action="store_true")

    args = ap.parse_args()

    y_cols = [c.strip() for c in args.y_cols.split(",") if c.strip()]
    if not y_cols:
        raise ValueError("--y_cols must contain at least one column name")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_train = pd.read_csv(args.train_csv)
    raw_test = pd.read_csv(args.test_csv)

    print("\nMissing labels in RAW TRAIN:")
    print(raw_train[y_cols].isna().sum().to_string())

    print("\nMissing labels in RAW TEST:")
    print(raw_test[y_cols].isna().sum().to_string())

    train_clean = process_split(
        raw_train,
        split_name="TRAIN raw -> cleaned",
        y_cols=y_cols,
        log10=args.log10,
    )

    test_clean = process_split(
        raw_test,
        split_name="TEST raw -> cleaned",
        y_cols=y_cols,
        log10=args.log10,
    )

    prefix = make_output_prefix(args.out_prefix, args.log10)

    train_out = out_dir / f"{prefix}_train.csv"
    test_out = out_dir / f"{prefix}_test.csv"

    train_clean.to_csv(train_out, index=False)
    test_clean.to_csv(test_out, index=False)

    print("\nWrote:")
    print(" ", train_out, "shape=", train_clean.shape)
    print(" ", test_out, "shape=", test_clean.shape)


if __name__ == "__main__":
    main()
