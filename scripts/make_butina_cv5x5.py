import os
import pandas as pd
import argparse
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


def mol_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles)


def morgan_fp(mol, radius=2, n_bits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def make_butina_clusters(smiles_list, cutoff):
    mols = [mol_from_smiles(smi) for smi in smiles_list]
    fps = [morgan_fp(mol) for mol in mols]

    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - sim for sim in sims])

    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    return clusters


def make_cluster_cv_assignments(input_csv, out_splits_csv, n_folds, n_repeats, cutoff):
    df = pd.read_csv(input_csv)

    # Stable row ids for the master dataset
    df = df.copy()
    df["row_id"] = list(range(len(df)))

    # Compute Butina clusters per row (aligned with row_id)
    smiles_list = df["SMILES"].tolist()
    clusters = make_butina_clusters(smiles_list, cutoff=cutoff)

    # Map cluster_id -> list of row_ids
    cluster_to_rows = {}
    for cluster_id, cluster_row_ids in enumerate(clusters):
        cluster_to_rows[cluster_id] = list(cluster_row_ids)

    # Store cluster_id per row in the dataframe
    df["cluster_id"] = -1
    for cluster_id, cluster_row_ids in cluster_to_rows.items():
        for rid in cluster_row_ids:
            df.loc[rid, "cluster_id"] = cluster_id

    unique_clusters = sorted(cluster_to_rows.keys())

    rows = []

    # each time this loop runs, we get a 5-CV, so default args will give 5x5-CV
    # it is n_repeats that is controlling the amount of shuffling we do
    # if n_repeats was 1, this would just give a 5-CV
    for cv_iter in range(n_repeats):
        # Shuffle clusters differently each repeat
        cluster_df = pd.DataFrame({"cluster_id": unique_clusters})
        cluster_df = cluster_df.sample(frac=1.0, random_state=cv_iter).reset_index(
            drop=True
        )
        shuffled_clusters = cluster_df["cluster_id"].tolist()

        # We will assign clusters to folds trying to balance total molecule count per fold
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

        for cluster_id in shuffled_clusters:
            # cluster_rows = list of row_ids that belong to the cluster, not the cluster ID
            cluster_rows = cluster_to_rows[cluster_id]
            cluster_size = len(cluster_rows)

            # example, say this is true:
            # cluster_to_rows = {
            #     0: [0, 1],
            #     1: [2, 4],
            #     2: [3, 5],
            # }

            # then,
            # cluster_rows = cluster_to_rows[0]   # -> [0, 1]
            # cluster_size = len(cluster_rows)    # -> 2   (two molecules)

            # Put this cluster's row_ids into the fold that currently has the fewest molecules
            best_fold = min(fold_sizes, key=fold_sizes.get)
            # example, say this is true:
            # fold_sizes = {0: 10, 1: 7, 2: 7, 3: 9, 4: 12}
            # min(fold_sizes, key=fold_sizes.get)  # returns 1 (or 2, but ties resolve by order)

            # flatten list, so if this is true:
            # cluster_rows = cluster_to_rows[0]  -> [0, 1]
            # we'd have fold_row_ids[1].extend([0,1])
            # adding 0 and 1 to the fold_row_ids list, if it was an empty list, it would be [0,1]
            fold_row_ids[best_fold].extend(cluster_rows)
            # now we increment how many row_ids are in the best_fold by 2, as we added two molecules
            # this will change which fold is the best for us to put row_ids in the future
            # as we are trying to populate folds evenly using clusters
            fold_sizes[best_fold] += cluster_size

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

    return df.drop(columns=["cluster_id"]), clusters, splits


def sanity_check_splits(df, clusters, splits_df, n_folds, n_repeats):
    N = len(df)
    n_unique_clusters = len(clusters)

    print("\n==============================")
    print("Sanity check: Butina cluster CV 5x5")
    print("==============================")
    print("N molecules:", N)
    print("Unique clusters:", n_unique_clusters)
    print("n_repeats:", n_repeats, "n_folds:", n_folds)

    if n_unique_clusters < n_folds:
        print(
            "WARNING: fewer unique clusters than folds. Cannot do GroupKFold-style splitting."
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

    # Fold sizes and cluster overlap checks
    rowid_to_cluster = {}

    # data structure looks like this:
    #     rowid_to_cluster = {
    #   0: cluster_id_for_row_0,
    #   1: cluster_id_for_row_1,
    #   2: cluster_id_for_row_2,
    #   ...
    #   N-1: cluster_id_for_row_N_minus_1
    # }
    # we can just do rowid_to_cluster[137] to get the cluster_id for that row_id
    # which we do in the following code
    for cluster_id, cluster_row_ids in enumerate(clusters):
        for rid in cluster_row_ids:
            rowid_to_cluster[rid] = cluster_id

    print("\nValidation fold sizes (rows and % of dataset):")
    for cv_iter in range(n_repeats):
        for fold in range(n_folds):
            val_ids = splits_df[
                (splits_df["cv_iter"] == cv_iter) & (splits_df["fold"] == fold)
            ]["row_id"].tolist()

            val_size = len(val_ids)
            val_pct = 100.0 * val_size / N
            val_clusters = set(rowid_to_cluster[rid] for rid in val_ids)

            train_ids = splits_df[splits_df["cv_iter"] == cv_iter]
            train_ids = train_ids[train_ids["fold"] != fold]["row_id"].tolist()
            train_clusters = set(rowid_to_cluster[rid] for rid in train_ids)

            overlap = val_clusters.intersection(train_clusters)

            print(
                f"  repeat {cv_iter}, fold {fold}: "
                f"val_rows={val_size} ({val_pct:.1f}%), "
                f"val_clusters={len(val_clusters)}, "
                f"cluster_overlap_with_train={len(overlap)}"
            )

            if len(overlap) != 0:
                print("    WARNING: cluster overlap found (should be 0).")

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
    ap.add_argument("--butina_dist_cutoff", type=float, required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_splits), exist_ok=True)
    os.makedirs(args.out_cv_root, exist_ok=True)

    df, clusters, splits = make_cluster_cv_assignments(
        args.input_csv,
        args.out_splits,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        cutoff=args.butina_dist_cutoff,
    )

    sanity_check_splits(
        df,
        clusters,
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
    )


if __name__ == "__main__":
    main()
