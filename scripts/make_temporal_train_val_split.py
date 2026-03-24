import os
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--train_frac", type=float, default=0.8)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv).copy()

    if not 0 < args.train_frac < 1:
        raise ValueError("--train_frac must be between 0 and 1")

    n_rows = len(df)
    n_train = int(args.train_frac * n_rows)
    n_val = n_rows - n_train

    if n_train == 0:
        raise ValueError("Train split would be empty")

    if n_val == 0:
        raise ValueError("Validation split would be empty")

    df["split"] = "train"
    df.loc[df.index[n_train:], "split"] = "val"

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.out_csv, index=False)

    print(f"Read {args.in_csv} with {n_rows} rows")
    print(f"Assigned first {n_train} rows to train")
    print(f"Assigned last {n_val} rows to val")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
