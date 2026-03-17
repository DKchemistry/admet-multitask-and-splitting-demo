import os
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def make_trainval_csv(train_csv, val_csv, out_csv):
    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)

    df = pd.concat([train, val], axis=0, ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")

def make_scaffold_cv_assignments(trainval_csv, out_splits_csv, n_folds=5, n_repeats=5):
    df = pd.read_csv(trainval_csv)

    # Stable row ids for the master dataset
    df = df.copy()
    df["row_id"] = list(range(len(df)))

    # Compute scaffold per row (aligned with row_id)
    scaffolds = [murcko_scaffold(smi) for smi in df["SMILES"].tolist()]
    df["scaffold"] = scaffolds

    # Map scaffold -> list of row_ids
    scaffold_to_rows = {}
    for rid, scaff in zip(df["row_id"].tolist(), df["scaffold"].tolist()):
        scaffold_to_rows.setdefault(scaff, []).append(rid)

    unique_scaffolds = sorted(scaffold_to_rows.keys())

    rows = []

    # each time this loop runs, we get a 5-CV, so default args will give 5x5-CV
    # it is n_repeats that is controlling the amount of shuffling we do
    # if n_repeats was 1, this would just give a 5-CV
    for cv_iter in range(n_repeats):
        # Shuffle scaffolds differently each repeat
        scaff_df = pd.DataFrame({"scaffold": unique_scaffolds})
        scaff_df = scaff_df.sample(frac=1.0, random_state=cv_iter).reset_index(
            drop=True
        )
        shuffled_scaffolds = scaff_df["scaffold"].tolist()

        # We will assign scaffolds to folds trying to balance total molecule count per fold
        # This is also how the amount folds we make is actually handled, as we read n_folds here
        fold_row_ids = {fold: [] for fold in range(n_folds)}
        fold_sizes = {fold: 0 for fold in range(n_folds)}
        # On init, looks like:
        # fold_row_ids =
        # {
        #   0: [],
        #   1: [],
        #   2: [],
        #   3: [],
        #   4: [],
        # }

        # fold_sizes =
        # {
        #   0: 0,
        #   1: 0,
        #   2: 0,
        #   3: 0,
        #   4: 0,
        # }

        for scaff in shuffled_scaffolds:
            # scaffold_rows = list of row_ids that belong to the scaffold, not the scaffold ID
            scaffold_rows = scaffold_to_rows[scaff]
            scaffold_size = len(scaffold_rows)

            # example, say this is true:
            # scaffold_to_rows = {
            #     "S1": [0, 1],
            #     "S2": [2, 4],
            #     "S3": [3, 5],
            # }

            # then,
            # scaffold_rows = scaffold_to_rows["S1"]   # -> [0, 1]
            # scaffold_size = len(scaffold_rows)      # -> 2   (two molecules)

            # Put this scaffold's row_ids into the fold that currently has the fewest molecules
            best_fold = min(fold_sizes, key=fold_sizes.get)
            # example, say this is true:
            # fold_sizes = {0: 10, 1: 7, 2: 7, 3: 9, 4: 12}
            # min(fold_sizes, key=fold_sizes.get)  # returns 1 (or 2, but ties resolve by order)

            # flatten list, so if this is true:
            # scaffold_rows = scaffold_to_rows["S1"]  -> [0, 1]
            # we'd have fold_row_ids[1].extend([0,1])
            # adding 0 and 1 to the fold_row_ids list, if it was an empty list, it would be [0,1]
            fold_row_ids[best_fold].extend(scaffold_rows)
            # now we increment how many row_ids are in the best_fold by 2, as we added two molecules
            # this will change which fold is the best for us to put row_ids in the future
            # as we are trying to populate folds evenly (using scaffolds, not mols, so careful about how your data looks)
            fold_sizes[best_fold] += scaffold_size

        # Record row_id -> fold for this repeat
        for fold in range(n_folds):
            for rid in fold_row_ids[fold]:
                # list of dicts
                rows.append({"row_id": rid, "cv_iter": cv_iter, "fold": fold})

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

    return df.drop(columns=["scaffold"]), scaffolds, splits


def sanity_check_splits(df, scaffolds, splits_df, n_folds=5, n_repeats=5):
    N = len(df)
    n_unique_scaffolds = len(set(scaffolds))

    print("\n==============================")
    print("Sanity check: scaffold CV 5x5")
    print("==============================")
    print("N molecules:", N)
    print("Unique scaffolds:", n_unique_scaffolds)
    print("n_repeats:", n_repeats, "n_folds:", n_folds)

    if n_unique_scaffolds < n_folds:
        print(
            "WARNING: fewer unique scaffolds than folds. Cannot do GroupKFold-style splitting."
        )
        return

    # Each repeat should assign every row_id exactly once
    for cv_iter in range(n_repeats):
        sub = splits_df[splits_df["cv_iter"] == cv_iter]
        counts = sub["row_id"].value_counts()

        n_missing = N - counts.size
        n_duplicates = (counts > 1).sum()

        if n_missing == 0 and n_duplicates == 0:
            print(f"Repeat {cv_iter}: OK (each row_id appears exactly once)")
        else:
            print(f"Repeat {cv_iter}: PROBLEM")
            print("  Missing row_ids:", n_missing)
            print("  Duplicated row_ids:", n_duplicates)

    # Fold sizes and scaffold overlap checks
    rowid_to_scaffold = {i: scaffolds[i] for i in range(N)}

    # data structure looks like this:
    #     rowid_to_scaffold = {
    #   0: scaffolds[0],
    #   1: scaffolds[1],
    #   2: scaffolds[2],
    #   ...
    #   N-1: scaffolds[N-1]
    # }
    # we can just do rowid_to_scaffold[137] to get the scaffold for that row_id
    # which we in the following code

    print("\nValidation fold sizes (rows and % of dataset):")
    for cv_iter in range(n_repeats):
        for fold in range(n_folds):
            val_ids = splits_df[
                (splits_df["cv_iter"] == cv_iter) & (splits_df["fold"] == fold)
            ]["row_id"].tolist()

            val_size = len(val_ids)
            val_pct = 100.0 * val_size / N
            val_scaffolds = set(rowid_to_scaffold[rid] for rid in val_ids)

            train_ids = splits_df[splits_df["cv_iter"] == cv_iter]
            train_ids = train_ids[train_ids["fold"] != fold]["row_id"].tolist()
            train_scaffolds = set(rowid_to_scaffold[rid] for rid in train_ids)

            overlap = val_scaffolds.intersection(train_scaffolds)

            print(
                f"  repeat {cv_iter}, fold {fold}: "
                f"val_rows={val_size} ({val_pct:.1f}%), "
                f"val_scaffolds={len(val_scaffolds)}, "
                f"scaffold_overlap_with_train={len(overlap)}"
            )

            if len(overlap) != 0:
                print("    WARNING: scaffold overlap found (should be 0).")

    # Check that repeats actually differ
    print("\nRepeat-to-repeat overlap check (are fold assignments changing?)")
    for fold in range(n_folds):
        base = set(
            splits_df[(splits_df["cv_iter"] == 0) & (splits_df["fold"] == fold)][
                "row_id"
            ]
        )
        for cv_iter in range(1, n_repeats):
            other = set(
                splits_df[
                    (splits_df["cv_iter"] == cv_iter) & (splits_df["fold"] == fold)
                ]["row_id"]
            )
            jaccard = len(base.intersection(other)) / len(base.union(other))
            print(f"  fold {fold}: repeat0 vs repeat{cv_iter} Jaccard={jaccard:.3f}")


def write_cv_fold_csvs(trainval_csv, splits_df, out_root, n_repeats=5, n_folds=5):
    """
    Write one CSV per (cv_iter, fold) containing the full dataset plus a 'split' column.

    Chemprop CLI can then use:
      --data-path <that_csv>
      --splits-column split

    split values are: 'train' or 'val'
    """
    df = pd.read_csv(trainval_csv).copy()

    # Ensure row_id is present and matches the split generator convention
    # this is going to run every time, doesn't really make sense to have this as an if statement
    if "row_id" not in df.columns:
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
    root = os.path.dirname(os.path.dirname(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_csv",
        default=os.path.join(root, "data/processed/caco2_train_paired_temporal.csv"),
    )
    ap.add_argument(
        "--val_csv",
        default=os.path.join(root, "data/processed/caco2_val_paired_temporal.csv"),
    )
    ap.add_argument(
        "--out_trainval",
        default=os.path.join(root, "data/processed/caco2_trainval_paired.csv"),
    )
    ap.add_argument(
        "--out_splits",
        default=os.path.join(root, "data/splits/caco2_scaffold_cv5x5.csv"),
    )
    ap.add_argument("--out_cv_root", default=os.path.join(root, "data/cv5x5"))
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_splits), exist_ok=True)

    make_trainval_csv(args.train_csv, args.val_csv, args.out_trainval)

    # already default but doesn't hurt
    n_folds = args.n_folds
    n_repeats = args.n_repeats

    df, scaffolds, splits = make_scaffold_cv_assignments(
        args.out_trainval, args.out_splits, n_folds=n_folds, n_repeats=n_repeats
    )

    sanity_check_splits(df, scaffolds, splits, n_folds=n_folds, n_repeats=n_repeats)

    write_cv_fold_csvs(
        args.out_trainval,
        splits,
        args.out_cv_root,
        n_repeats=n_repeats,
        n_folds=n_folds,
    )


if __name__ == "__main__":
    main()
