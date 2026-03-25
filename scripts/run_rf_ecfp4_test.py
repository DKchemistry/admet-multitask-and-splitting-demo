import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau

RDLogger.DisableLog("rdApp.warning")

TRAIN = "data/processed/hlm_mlm_paired_log10_train.csv"
TEST = "data/processed/hlm_mlm_paired_log10_test.csv"
RESULTS_DIR = "results/hlm_mlm_test_rf_ecfp4"

def ecfp4_from_smiles(smiles_list, n_bits=2048):
    fps = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        fps.append(list(fp))

    return fps

def fit_and_predict(train_path, test_path):

    # SMILES -> ECFP4
    train_df = pd.read_csv(train_path)
    train_smiles = train_df["SMILES"].values.tolist()
    train_fps = ecfp4_from_smiles(train_smiles)

    test_df = pd.read_csv(test_path)
    test_smiles = test_df["SMILES"].values.tolist()
    test_fps = ecfp4_from_smiles(test_smiles)

    # HLM CLint
    train_hlm = train_df["HLM CLint"].values.tolist()
    regr_hlm = RandomForestRegressor(n_estimators=500, n_jobs=-1)
    regr_hlm.fit(train_fps, train_hlm)
    hlm_pred = regr_hlm.predict(test_fps)

    # Predictions & Metrics
    hlm = test_df["HLM CLint"].values.tolist()

    hlm_pred_df = pd.DataFrame({"SMILES":test_smiles, "HLM CLint": hlm, "HLM CLint pred": hlm_pred})
    print(hlm_pred_df.head())
    os.makedirs(RESULTS_DIR, exist_ok=True)
    hlm_pred_path = os.path.join(RESULTS_DIR, "hlm_predictions.csv")
    hlm_pred_df.to_csv(hlm_pred_path)
    print(f"Wrote f{hlm_pred_path}")

    rmse = mean_squared_error(hlm, hlm_pred) ** 2
    mae = mean_absolute_error(hlm, hlm_pred)
    r2 = r2_score(hlm, hlm_pred)
    tau, tau_pvalue = kendalltau(hlm, hlm_pred)

    hlm_metrics_df = pd.DataFrame(
        [
            {"target": "HLM CLint",
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "kendall_tau": tau,
            "kendall_tau_pvalue": tau_pvalue,
            }
        ]
    )
    print(hlm_metrics_df)

    # MLM CLint
    train_mlm = train_df["MLM CLint"].values.tolist()
    regr_mlm = RandomForestRegressor(n_estimators=500, n_jobs=-1)
    regr_mlm.fit(train_fps, train_mlm)
    mlm_pred = regr_mlm.predict(test_fps)

    # Predictions & Metrics
    mlm = test_df["MLM CLint"].values.tolist()

    mlm_pred_df = pd.DataFrame(
        {"SMILES": test_smiles, "MLM CLint": mlm, "MLM CLint pred": mlm_pred}
    )
    print(mlm_pred_df.head())
    mlm_pred_path = os.path.join(RESULTS_DIR, "mlm_predictions.csv")
    mlm_pred_df.to_csv(mlm_pred_path)
    print(f"Wrote f{mlm_pred_path}")

    # Safe to overwrite as we wrote previous metrics to df
    rmse = mean_squared_error(mlm, mlm_pred) ** 2
    mae = mean_absolute_error(mlm, mlm_pred)
    r2 = r2_score(mlm, mlm_pred)
    tau, tau_pvalue = kendalltau(mlm, mlm_pred)

    mlm_metrics_df = pd.DataFrame(
        [
            {
                "target": "MLM CLint",
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "kendall_tau": tau,
                "kendall_tau_pvalue": tau_pvalue,
            }
        ]
    )
    print(mlm_metrics_df)

    combined_metrics_df = pd.concat([hlm_metrics_df, mlm_metrics_df])
    print(combined_metrics_df)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    combined_metrics_path = os.path.join(RESULTS_DIR, "metrics_all_targets.csv")
    combined_metrics_df.to_csv(combined_metrics_path)
    print(f"Wrote {combined_metrics_path}")

def main(): 
    fit_and_predict(train_path=TRAIN, test_path=TEST)

if __name__ == "__main__":
    main()
