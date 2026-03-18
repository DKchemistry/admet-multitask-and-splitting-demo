import os
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def make_scaffold_cv_assignments(input_csv, out_splits_csv, n_folds, n_repeats):
    df = pd.read_csv(input_csv)

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


def sanity_check_splits(df, scaffolds, splits_df, n_folds, n_repeats):
    N = len(df)
    n_unique_scaffolds = len(set(scaffolds))

    print("\n==============================")
    print("Sanity check: scaffold outer CV 5x5")
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

    print("\nOuter test fold sizes (rows and % of dataset):")
    for cv_iter in range(n_repeats):
        for fold in range(n_folds):
            test_ids = splits_df[
                (splits_df["cv_iter"] == cv_iter) & (splits_df["fold"] == fold)
            ]["row_id"].tolist()

            test_size = len(test_ids)
            test_pct = 100.0 * test_size / N
            test_scaffolds = set(rowid_to_scaffold[rid] for rid in test_ids)

            train_ids = splits_df[splits_df["cv_iter"] == cv_iter]
            train_ids = train_ids[train_ids["fold"] != fold]["row_id"].tolist()
            train_scaffolds = set(rowid_to_scaffold[rid] for rid in train_ids)

            overlap = test_scaffolds.intersection(train_scaffolds)

            print(
                f"  repeat {cv_iter}, fold {fold}: "
                f"test_rows={test_size} ({test_pct:.1f}%), "
                f"test_scaffolds={len(test_scaffolds)}, "
                f"scaffold_overlap_with_outer_train={len(overlap)}"
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


def sanity_check_written_fold_csvs(out_root, n_repeats, n_folds):
    print("\n==============================")
    print("Sanity check: written train/val/test fold CSVs")
    print("==============================")

    for cv_iter in range(n_repeats):
        for fold in range(n_folds):
            path = os.path.join(out_root, f"iter_{cv_iter}", f"fold_{fold}.csv")
            df = pd.read_csv(path)

            split_counts = df["split"].value_counts().to_dict()
            print(f"iter {cv_iter} fold {fold}: {split_counts}")

            # Every row should have exactly one of train/val/test
            bad_labels = set(df["split"].unique()) - {"train", "val", "test"}
            if bad_labels:
                raise ValueError(
                    f"Unexpected split labels in {path}: {sorted(bad_labels)}"
                )

            # No row_id duplication
            if df["row_id"].duplicated().any():
                raise ValueError(f"Duplicated row_id found in {path}")

            # Check inner scaffold separation: val scaffolds should not overlap train scaffolds
            scaffolds = [murcko_scaffold(smi) for smi in df["SMILES"].tolist()]
            df = df.copy()
            df["scaffold"] = scaffolds

            train_scaff = set(df.loc[df["split"] == "train", "scaffold"])
            val_scaff = set(df.loc[df["split"] == "val", "scaffold"])
            test_scaff = set(df.loc[df["split"] == "test", "scaffold"])

            if train_scaff.intersection(val_scaff):
                raise ValueError(
                    f"Inner train/val scaffold overlap in iter {cv_iter} fold {fold}"
                )

            if test_scaff.intersection(val_scaff):
                raise ValueError(
                    f"Val/test scaffold overlap in iter {cv_iter} fold {fold}"
                )

            if test_scaff.intersection(train_scaff):
                raise ValueError(
                    f"Train/test scaffold overlap in iter {cv_iter} fold {fold}"
                )


def assign_inner_scaffold_val_ids(
    outer_train_df,
    val_frac=0.20,
    random_state=0,
):
    """
    Split the outer-training rows into inner train/val by scaffold.

    We keep whole scaffolds together and try to make the inner validation set
    roughly val_frac of the outer-training pool.
    """
    work = outer_train_df.copy()

    scaffold_to_rows = {}
    for rid, scaff in zip(work["row_id"].tolist(), work["scaffold"].tolist()):
        scaffold_to_rows.setdefault(scaff, []).append(rid)

    scaffolds = sorted(scaffold_to_rows.keys())

    # Shuffle scaffold order so different repeats/folds can differ reproducibly
    scaff_df = pd.DataFrame({"scaffold": scaffolds})
    scaff_df = scaff_df.sample(frac=1.0, random_state=random_state).reset_index(
        drop=True
    )
    shuffled_scaffolds = scaff_df["scaffold"].tolist()

    target_val_size = int(round(len(work) * val_frac))
    val_ids = []
    val_size = 0

    # Greedy fill of the inner validation set using whole scaffolds
    for scaff in shuffled_scaffolds:
        scaffold_rows = scaffold_to_rows[scaff]
        scaffold_size = len(scaffold_rows)

        # If val is already at or above target, stop
        if val_size >= target_val_size:
            break

        val_ids.extend(scaffold_rows)
        val_size += scaffold_size

    return sorted(val_ids)


def write_cv_fold_csvs(
    input_csv,
    splits_df,
    out_root,
    n_repeats,
    n_folds,
    inner_val_frac=0.20,
):
    """
    Write one CSV per (cv_iter, fold) containing the full dataset plus a 'split' column.

    split values are:
      - 'train' : inner training rows
      - 'val'   : inner validation rows used for early stopping
      - 'test'  : outer held-out fold used for final scoring in that CV run
    """
    df = pd.read_csv(input_csv).copy()
    df["row_id"] = list(range(len(df)))
    df["scaffold"] = [murcko_scaffold(smi) for smi in df["SMILES"].tolist()]

    os.makedirs(out_root, exist_ok=True)

    for cv_iter in range(n_repeats):
        iter_dir = os.path.join(out_root, f"iter_{cv_iter}")
        os.makedirs(iter_dir, exist_ok=True)

        for fold in range(n_folds):
            # Outer test fold = the old held-out fold from 5-fold CV
            test_ids = splits_df[
                (splits_df["cv_iter"] == cv_iter) & (splits_df["fold"] == fold)
            ]["row_id"].tolist()
            test_id_set = set(test_ids)

            # Outer training pool = everything not in outer test
            outer_train_df = df[~df["row_id"].isin(test_id_set)].copy()

            # Inner validation split is created ONLY from the outer training pool
            # We vary the random_state by repeat and fold for reproducibility.
            inner_seed = 10_000 * cv_iter + fold
            val_ids = assign_inner_scaffold_val_ids(
                outer_train_df,
                val_frac=inner_val_frac,
                random_state=inner_seed,
            )
            val_id_set = set(val_ids)

            # Sanity: val and test must be disjoint
            overlap = val_id_set.intersection(test_id_set)
            if overlap:
                raise ValueError(
                    f"Inner val / outer test overlap found for iter {cv_iter}, fold {fold}"
                )

            out = df.drop(columns=["scaffold"]).copy()
            out["split"] = "train"
            out.loc[out["row_id"].isin(val_id_set), "split"] = "val"
            out.loc[out["row_id"].isin(test_id_set), "split"] = "test"

            out_path = os.path.join(iter_dir, f"fold_{fold}.csv")
            out.to_csv(out_path, index=False)

            n_train = (out["split"] == "train").sum()
            n_val = (out["split"] == "val").sum()
            n_test = (out["split"] == "test").sum()

            print(
                f"iter {cv_iter} fold {fold}: "
                f"train={n_train} val={n_val} test={n_test}"
            )

    print(f"Wrote per-fold CV CSVs under: {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_splits", required=True)
    ap.add_argument("--out_cv_root", required=True)
    ap.add_argument("--n_folds", type=int, required=True)
    ap.add_argument("--n_repeats", type=int, required=True)
    ap.add_argument("--inner_val_frac", type=float, default=0.20)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_splits), exist_ok=True)
    os.makedirs(args.out_cv_root, exist_ok=True)

    df, scaffolds, splits = make_scaffold_cv_assignments(
        args.input_csv,
        args.out_splits,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
    )

    sanity_check_splits(
        df,
        scaffolds,
        splits,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
    )

    write_cv_fold_csvs(
        args.input_csv,
        splits,
        args.out_cv_root,
        n_repeats=args.n_repeats,
        n_folds=args.n_folds,
        inner_val_frac=args.inner_val_frac,
    )

    sanity_check_written_fold_csvs(
        args.out_cv_root,
        n_repeats=args.n_repeats,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
