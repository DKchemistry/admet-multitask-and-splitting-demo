import os
import argparse
import pandas as pd


def prepare_inner_split(
    input_csv, out_trainval_csv, out_test_csv, val_frac=0.2, seed=0
):
    df = pd.read_csv(input_csv).copy()

    if "split" not in df.columns:
        raise ValueError(f"Expected a 'split' column in {input_csv}")

    outer_train = df[df["split"] == "train"].copy()
    outer_test = df[df["split"] == "val"].copy()

    if len(outer_train) == 0:
        raise ValueError(f"No outer-train rows found in {input_csv}")
    if len(outer_test) == 0:
        raise ValueError(f"No outer-test rows found in {input_csv}")

    inner_val = outer_train.sample(frac=val_frac, random_state=seed)
    inner_val_ids = set(inner_val["row_id"].tolist())

    trainval = outer_train.copy()
    trainval["split"] = "train"
    trainval.loc[trainval["row_id"].isin(inner_val_ids), "split"] = "val"

    test = outer_test.copy()

    os.makedirs(os.path.dirname(out_trainval_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_test_csv), exist_ok=True)

    trainval.to_csv(out_trainval_csv, index=False)
    test.to_csv(out_test_csv, index=False)

    print(f"Wrote inner train/val file: {out_trainval_csv} shape={trainval.shape}")
    print(f"Wrote outer test file    : {out_test_csv} shape={test.shape}")
    print(f"Inner split counts:\n{trainval['split'].value_counts()}")
    print(f"Outer test rows: {len(test)}")


def prepare_tree(input_fold_root, output_root, val_frac=0.2, seed=0):
    if not os.path.isdir(input_fold_root):
        raise ValueError(f"Input fold root does not exist: {input_fold_root}")

    iter_dirs = sorted(
        d
        for d in os.listdir(input_fold_root)
        if d.startswith("iter_") and os.path.isdir(os.path.join(input_fold_root, d))
    )

    if not iter_dirs:
        raise ValueError(f"No iter_* directories found under {input_fold_root}")

    n_written = 0

    for iter_dir in iter_dirs:
        iter_path = os.path.join(input_fold_root, iter_dir)

        fold_files = sorted(
            f
            for f in os.listdir(iter_path)
            if f.startswith("fold_") and f.endswith(".csv")
        )

        if not fold_files:
            raise ValueError(f"No fold_*.csv files found under {iter_path}")

        for fold_file in fold_files:
            input_csv = os.path.join(iter_path, fold_file)

            fold_name = fold_file.replace(".csv", "")
            out_dir = os.path.join(output_root, iter_dir, fold_name)

            out_trainval_csv = os.path.join(out_dir, "trainval.csv")
            out_test_csv = os.path.join(out_dir, "test.csv")

            print("\n==============================")
            print(f"Preparing {input_csv}")
            print(f"Output dir: {out_dir}")
            print("==============================")

            prepare_inner_split(
                input_csv=input_csv,
                out_trainval_csv=out_trainval_csv,
                out_test_csv=out_test_csv,
                val_frac=val_frac,
                seed=seed,
            )

            n_written += 1

    print("\nDone.")
    print(f"Prepared {n_written} run directories under: {output_root}")


def main():
    ap = argparse.ArgumentParser()

    # one-file mode
    ap.add_argument("--input_csv")
    ap.add_argument("--out_trainval_csv")
    ap.add_argument("--out_test_csv")

    # whole-tree mode
    ap.add_argument("--input_fold_root")
    ap.add_argument("--output_root")

    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    one_file_mode = (
        args.input_csv is not None
        and args.out_trainval_csv is not None
        and args.out_test_csv is not None
    )

    whole_tree_mode = args.input_fold_root is not None and args.output_root is not None

    if one_file_mode and whole_tree_mode:
        raise ValueError("Choose either one-file mode or whole-tree mode, not both.")

    if one_file_mode:
        prepare_inner_split(
            input_csv=args.input_csv,
            out_trainval_csv=args.out_trainval_csv,
            out_test_csv=args.out_test_csv,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        return

    if whole_tree_mode:
        prepare_tree(
            input_fold_root=args.input_fold_root,
            output_root=args.output_root,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        return

    raise ValueError(
        "Provide either:\n"
        "  --input_csv --out_trainval_csv --out_test_csv\n"
        "or:\n"
        "  --input_fold_root --output_root"
    )


if __name__ == "__main__":
    main()
