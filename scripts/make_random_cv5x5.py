import os
import pandas as pd
import argparse


def make_random_cv_assignments(input_csv, out_splits_csv, n_folds, n_repeats):
    df = pd.read_csv(input_csv)

    # Stable row ids for the master dataset
    df = df.copy()
    df["row_id"] = list(range(len(df)))

    row_ids = df["row_id"].tolist()
    rows = []

    # each time this loop runs, we get a 5-CV, so default args will give 5x5-CV
    # it is n_repeats that is controlling the amount of shuffling we do
    # if n_repeats was 1, this would just give a 5-CV
    for cv_iter in range(n_repeats):
        # Shuffle row_ids differently each repeat
        shuffled = (
            pd.DataFrame({"row_id": row_ids})
            .sample(frac=1.0, random_state=cv_iter)
            .reset_index(drop=True)
        )

        # Split the shuffled row_ids into n_folds chunks
        fold_sizes = [len(df) // n_folds] * n_folds
        for i in range(len(df) % n_folds):
            fold_sizes[i] += 1

        start = 0
        for fold in range(n_folds):
            stop = start + fold_sizes[fold]
            fold_row_ids = shuffled.iloc[start:stop]["row_id"].tolist()

            for rid in fold_row_ids:
                rows.append({"row_id": rid, "cv_iter": cv_iter, "fold": fold})

            start = stop

    # takes that list-of-dicts and makes a table with columns inferred from the dict keys ("cv_iter", "fold", "row_id")
    splits = (
        pd.DataFrame(rows)
        .sort_values(["cv_iter", "fold", "row_id"])
        .reset_index(drop=True)
    )
    splits.to_csv(out_splits_csv, index=False)
    print(
        f"Wrote {out_splits_csv} with {len(splits)} assignments (should be N * {n_repeats})"
    )

    return splits


def write_cv_fold_csvs(input_csv, splits_df, out_root, n_repeats, n_folds):
    """
    Write one CSV per (cv_iter, fold) containing the full dataset plus a 'split' column.

    Chemprop CLI can then use:
      --data-path <that_csv>
      --splits-column split

    split values are: 'train' or 'val'
    """
    df = pd.read_csv(input_csv).copy()

    # Ensure row_id matches the split generator convention
    df["row_id"] = list(range(len(df)))

    os.makedirs(out_root, exist_ok=True)

    for cv_iter in range(n_repeats):
        iter_dir = os.path.join(out_root, f"iter_{cv_iter}")
        os.makedirs(iter_dir, exist_ok=True)

        for fold in range(n_folds):
            val_ids = splits_df[
                (splits_df["cv_iter"] == cv_iter) & (splits_df["fold"] == fold)
            ]["row_id"].tolist()

            out = df.copy()
            out["split"] = "train"
            out.loc[out["row_id"].isin(val_ids), "split"] = "val"

            out_path = os.path.join(iter_dir, f"fold_{fold}.csv")
            out.to_csv(out_path, index=False)

    print(f"Wrote per-fold CV CSVs under: {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_splits", required=True)
    ap.add_argument("--out_cv_root", required=True)
    ap.add_argument("--n_folds", type=int, required=True)
    ap.add_argument("--n_repeats", type=int, required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_splits), exist_ok=True)
    os.makedirs(args.out_cv_root, exist_ok=True)

    splits = make_random_cv_assignments(
        args.input_csv,
        args.out_splits,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
    )

    write_cv_fold_csvs(
        args.input_csv,
        splits,
        args.out_cv_root,
        n_repeats=args.n_repeats,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
