import os
import argparse
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SMILES_COL = "SMILES"
PAPP_COL = "Caco-2 Permeability Papp A>B"
EFFLUX_COL = "Caco-2 Permeability Efflux"


def ecfp4_bits(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize(df, n_bits=2048):
    """
    Featurize SMILES into ECFP4 bit vectors.

    We keep row_id as a column, because row_id is how we match folds from the split file.
    """
    fps = []
    keep_rows = []

    for i, row in df.iterrows():
        smi = row[SMILES_COL]
        fp = ecfp4_bits(smi, n_bits=n_bits)
        if fp is None:
            continue
        fps.append(fp)
        keep_rows.append(i)

    if len(fps) == 0:
        X = np.zeros((0, n_bits), dtype=np.int8)
    else:
        X = np.vstack(fps)

    out = df.loc[keep_rows].reset_index(drop=True).copy()
    return X, out


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def get_fold_data(df, X, splits_df, cv_iter, fold):
    """
    Materialize train/val subsets for one (cv_iter, fold).

    Validation set = rows whose row_id is assigned to this fold in this repeat.
    Training set = everything else.
    """
    val_row_ids = splits_df[
        (splits_df["cv_iter"] == cv_iter) & (splits_df["fold"] == fold)
    ]["row_id"].tolist()

    is_val = df["row_id"].isin(val_row_ids).values

    train_df = df[~is_val].reset_index(drop=True).copy()
    val_df = df[is_val].reset_index(drop=True).copy()

    X_train = X[~is_val]
    X_val = X[is_val]

    return train_df, val_df, X_train, X_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trainval_csv", default="data/processed/caco2_trainval_paired.csv"
    )
    ap.add_argument("--splits_csv", default="data/splits/caco2_scaffold_cv5x5.csv")
    ap.add_argument("--out_dir", default="results/cv5x5/rf_ecfp4")
    ap.add_argument(
        "--y_cols", default="Caco-2 Permeability Papp A>B,Caco-2 Permeability Efflux"
    )
    ap.add_argument("--n_bits", type=int, default=2048)
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    y_cols = [c.strip() for c in args.y_cols.split(",") if c.strip()]
    if len(y_cols) != 2:
        raise ValueError("--y_cols must be exactly two comma-separated column names")

    y1_col = y_cols[0]
    y2_col = y_cols[1]

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load master dataset and create stable row_id
    df = pd.read_csv(args.trainval_csv).copy()
    if "row_id" not in df.columns:
        df["row_id"] = list(range(len(df)))

    splits_df = pd.read_csv(args.splits_csv)

    # Featurize once
    print("Featurizing full trainval dataset...")
    X, df = featurize(df, n_bits=args.n_bits)

    # If featurize dropped rows (rare, but possible), remove missing row_ids from splits
    valid_row_ids = set(df["row_id"].tolist())
    splits_df = splits_df[splits_df["row_id"].isin(valid_row_ids)].reset_index(
        drop=True
    )

    n_repeats = args.n_repeats
    n_folds = args.n_folds

    # Fix RF randomness for reproducibility (single seed, since we are not sampling seeds here)
    rf_kwargs = dict(
        n_estimators=500,
        random_state=0,
        n_jobs=-1,
        min_samples_leaf=1,
    )

    all_rows = []

    for cv_iter in range(n_repeats):
        for fold in range(n_folds):
            print("\n==============================")
            print(f"RF baseline: repeat {cv_iter} / fold {fold}")
            print("==============================")

            train_df, val_df, X_train, X_val = get_fold_data(
                df, X, splits_df, cv_iter, fold
            )

            y_train_1 = train_df[y1_col].values
            y_train_2 = train_df[y2_col].values
            y_train_multi = train_df[[y1_col, y2_col]].values

            y_val_1 = val_df[y1_col].values
            y_val_2 = val_df[y2_col].values
            y_val_multi = val_df[[y1_col, y2_col]].values

            # Train single-task models
            rf_1 = RandomForestRegressor(**rf_kwargs)
            rf_1.fit(X_train, y_train_1)
            pred_val_1 = rf_1.predict(X_val)

            rf_2 = RandomForestRegressor(**rf_kwargs)
            rf_2.fit(X_train, y_train_2)
            pred_val_2 = rf_2.predict(X_val)

            # Train multi-output model
            rf_multi = RandomForestRegressor(**rf_kwargs)
            rf_multi.fit(X_train, y_train_multi)
            pred_val_multi = rf_multi.predict(X_val)

            # Metrics
            m_single_1 = metrics(y_val_1, pred_val_1)
            m_single_2 = metrics(y_val_2, pred_val_2)
            m_multi_1 = metrics(y_val_multi[:, 0], pred_val_multi[:, 0])
            m_multi_2 = metrics(y_val_multi[:, 1], pred_val_multi[:, 1])

            # Canonical long-form rows
            all_rows.append(
                {
                    "model_family": "rf",
                    "variant": f"single_{y1_col}",
                    "task": y1_col,
                    "cv_iter": cv_iter,
                    "fold": fold,
                    **m_single_1,
                }
            )
            all_rows.append(
                {
                    "model_family": "rf",
                    "variant": f"single_{y2_col}",
                    "task": y2_col,
                    "cv_iter": cv_iter,
                    "fold": fold,
                    **m_single_2,
                }
            )
            all_rows.append(
                {
                    "model_family": "rf",
                    "variant": "multi",
                    "task": y1_col,
                    "cv_iter": cv_iter,
                    "fold": fold,
                    **m_multi_1,
                }
            )
            all_rows.append(
                {
                    "model_family": "rf",
                    "variant": "multi",
                    "task": y2_col,
                    "cv_iter": cv_iter,
                    "fold": fold,
                    **m_multi_2,
                }
            )

            # Save predictions for just one fold for debugging/inspection
            if cv_iter == 0 and fold == 0:
                out_pred = val_df[[SMILES_COL, "row_id", y1_col, y2_col]].copy()
                out_pred["pred_single_1"] = pred_val_1
                out_pred["pred_single_2"] = pred_val_2
                out_pred["pred_multi_1"] = pred_val_multi[:, 0]
                out_pred["pred_multi_2"] = pred_val_multi[:, 1]
                out_pred.to_csv(
                    os.path.join(out_dir, "val_predictions_iter0_fold0.csv"),
                    index=False,
                )

    metrics_df = pd.DataFrame(all_rows)
    metrics_df = metrics_df.round({"mae": 3, "rmse": 3, "r2": 4})

    out_metrics = os.path.join(out_dir, "rf_metrics_long.csv")
    metrics_df.to_csv(out_metrics, index=False)

    # Optional summary (mean/std across the 25 folds)
    summary = metrics_df.groupby(["model_family", "variant", "task"], as_index=False)[
        ["mae", "rmse", "r2"]
    ].agg(["mean", "std"])
    summary.columns = [
        "_".join([c for c in col if c]) for col in summary.columns.values
    ]
    summary = summary.round(4)

    out_summary = os.path.join(out_dir, "rf_metrics_summary.csv")
    summary.to_csv(out_summary, index=False)

    print("\nWrote:")
    print(" -", out_metrics)
    print(" -", out_summary)
    print(" -", os.path.join(out_dir, "val_predictions_iter0_fold0.csv"))


if __name__ == "__main__":
    main()
