import os

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
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)  # ECFP4 = radius 2
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize(df, n_bits=2048):
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
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def main():
    train_path = "data/processed/caco2_train_paired_temporal.csv"
    val_path = "data/processed/caco2_val_paired_temporal.csv"

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    print("Featurizing train...")
    X_train, df_train = featurize(df_train, n_bits=2048)
    print("Featurizing val...")
    X_val, df_val = featurize(df_val, n_bits=2048)

    y_train_papp = df_train[PAPP_COL].values
    y_train_efflux = df_train[EFFLUX_COL].values
    y_train_multi = df_train[[PAPP_COL, EFFLUX_COL]].values

    y_val_papp = df_val[PAPP_COL].values
    y_val_efflux = df_val[EFFLUX_COL].values
    y_val_multi = df_val[[PAPP_COL, EFFLUX_COL]].values

    seeds = [0, 1, 2, 3, 4]
    all_rows = []

    for seed in seeds:
        print("\n==============================")
        print("Seed:", seed)
        print("==============================")

        rf_kwargs = dict(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
        )

        print("Training RF single-task: Papp...")
        rf_papp = RandomForestRegressor(**rf_kwargs)
        rf_papp.fit(X_train, y_train_papp)
        pred_val_papp = rf_papp.predict(X_val)

        print("Training RF single-task: Efflux...")
        rf_efflux = RandomForestRegressor(**rf_kwargs)
        rf_efflux.fit(X_train, y_train_efflux)
        pred_val_efflux = rf_efflux.predict(X_val)

        print("Training RF multi-output: (Papp, Efflux)...")
        rf_multi = RandomForestRegressor(**rf_kwargs)
        rf_multi.fit(X_train, y_train_multi)
        pred_val_multi = rf_multi.predict(X_val)

        m_single_papp = metrics(y_val_papp, pred_val_papp)
        m_single_efflux = metrics(y_val_efflux, pred_val_efflux)
        m_multi_papp = metrics(y_val_multi[:, 0], pred_val_multi[:, 0])
        m_multi_efflux = metrics(y_val_multi[:, 1], pred_val_multi[:, 1])

        print("\nValidation metrics (seed", seed, ")")
        print("  Single-task Papp   :", m_single_papp)
        print("  Single-task Efflux :", m_single_efflux)
        print("  Multi-task Papp    :", m_multi_papp)
        print("  Multi-task Efflux  :", m_multi_efflux)

        # Store metrics in a simple long-form table
        all_rows.append({"seed": seed, "model": "single", "task": "papp", **m_single_papp})
        all_rows.append({"seed": seed, "model": "single", "task": "efflux", **m_single_efflux})
        all_rows.append({"seed": seed, "model": "multi", "task": "papp", **m_multi_papp})
        all_rows.append({"seed": seed, "model": "multi", "task": "efflux", **m_multi_efflux})

        # Save per-sample predictions for seed 0 only 
        if seed == 0:
            out_pred = df_val[[SMILES_COL, PAPP_COL, EFFLUX_COL]].copy()
            out_pred["pred_single_papp"] = pred_val_papp
            out_pred["pred_single_efflux"] = pred_val_efflux
            out_pred["pred_multi_papp"] = pred_val_multi[:, 0]
            out_pred["pred_multi_efflux"] = pred_val_multi[:, 1]
            out_pred.to_csv(os.path.join(out_dir, "rf_val_predictions_seed0.csv"), index=False)

    metrics_df = pd.DataFrame(all_rows)
    metrics_df = metrics_df.round({"mae": 3, "rmse": 3, "r2": 4})
    metrics_path = os.path.join(out_dir, "rf_val_metrics_by_seed.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Mean/std summary for quick reporting
    summary = (
        metrics_df
        .groupby(["model", "task"], as_index=False)[["mae", "rmse", "r2"]]
        .agg(["mean", "std"])
    )
    # Flatten column names for readability
    summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.values]
    summary = summary.round(4)
    summary_path = os.path.join(out_dir, "rf_val_metrics_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nWrote:")
    print(" -", metrics_path)
    print(" -", summary_path)
    print(" - results/rf_val_predictions_seed0.csv")


if __name__ == "__main__":
    main()
