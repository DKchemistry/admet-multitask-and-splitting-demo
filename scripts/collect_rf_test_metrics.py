import os
import glob
import pathlib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau

RESULTS_CSVS = "results/hlm_mlm_test_rf_ecfp4/**/*lm_predictions.csv"
TEST = "data/processed/hlm_mlm_paired_log10_test.csv"
SUMMARY_DIR = "results/hlm_mlm_test_rf_ecfp4/summary"


def compute_metrics(y_true, y_pred, target):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    tau, tau_pvalue = kendalltau(y_true, y_pred)

    return {
        "target": target,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "kendall_tau": tau,
        "kendall_tau_pvalue": tau_pvalue,
    }


truth_df = pd.read_csv(TEST)

pred_csvs = []

for filename in glob.glob(RESULTS_CSVS):
    path = pathlib.Path(filename)
    seed_id = path.parent.name # e.g., seed_0
    target_pred = pd.read_csv(path)
    # header of hlm_predictions: SMILES,HLM CLint,HLM CLint pred
    # header of mlm_predictions: SMILES,MLM CLint,MLM CLint pred
    # We rename the pred column if it changes
    # and drop columns that could complicate merge

    if "HLM CLint pred" in target_pred.columns:
        hlm_seed_name = "HLM CLint pred" + "_" + seed_id
        target_pred.rename(columns={"HLM CLint pred":hlm_seed_name}, inplace=True)
        target_pred.drop(columns=["HLM CLint"], inplace=True)
    if "MLM CLint pred" in target_pred.columns:
        mlm_seed_name = "MLM CLint pred" + "_" + seed_id
        target_pred.rename(columns={"MLM CLint pred":mlm_seed_name}, inplace=True)
        target_pred.drop(columns=["MLM CLint"], inplace=True)
    # Now we should have:
    # SMILES, HLM CLint pred_seed_id
    # SMILES, MLM CLint pred_seed_id
    # So every col is unique except SMILES, so merge should be safe.
    pred_csvs.append(target_pred)

test_df = pd.read_csv(TEST)
for df in pred_csvs:
    test_df = test_df.merge(df, how="inner", on="SMILES")

# Now we want to get ensemble averages/std per target (HLM/MLM)
# Relevant columns will start w/ {target} and end with pred_seed_n
# These should be grouped for pandas operations
# A dictionary could work
# Two keys: HLM preds, MLM Preds
# Five vals per key

group_dict = {}

for col in test_df.columns:
    if "pred" in col and "HLM" in col:
        split_key = col.split("_")
        key = split_key[0]  # "HLM CLint pred"
        if key not in group_dict:
            group_dict[key] = []
        group_dict[key].append(col)
    if "pred" in col and "MLM" in col:
        split_key = col.split("_")
        key = split_key[0] # "MLM CLint pred"
        if key not in group_dict:
            group_dict[key] = []
        group_dict[key].append(col)

# print(
#     group_dict
# )  # {'HLM CLint pred': ['HLM CLint pred_seed_0', 'HLM CLint pred_seed_1', 'HLM CLint pred_seed_2', 'HLM CLint pred_seed_3', 'HLM CLint pred_seed_4'], 'MLM CLint pred': ['MLM CLint pred_seed_0', 'MLM CLint pred_seed_1', 'MLM CLint pred_seed_2', 'MLM CLint pred_seed_3', 'MLM CLint pred_seed_4']}

# Now we can use the dict to group operations in pandas

for keys in group_dict:
    ensemble_mean_col = str(keys)+"_ensemble_mean"
    ensemble_std_col = str(keys)+"_ensemble_std"
    test_df[ensemble_mean_col] = test_df[group_dict[keys]].mean(axis=1)
    test_df[ensemble_std_col] = test_df[group_dict[keys]].std(axis=1)

# print(test_df.columns) # getting both mean and std cols

# Now we have our per_molecule_predictions dataframe
os.makedirs(SUMMARY_DIR, exist_ok=True)
per_mol_pred_path = os.path.join(SUMMARY_DIR, "per_molecule_predictions.csv")
test_df.to_csv(per_mol_pred_path, index=False)
print(f"Wrote {per_mol_pred_path}")

# Now we need the ensemble_metrics dataframe
# We could do this with a loop but its overkill

hlm = test_df["HLM CLint"].values.tolist()
mlm = test_df["MLM CLint"].values.tolist() 

hlm_ens_pred = test_df["HLM CLint pred_ensemble_mean"].values.tolist()
mlm_ens_pred = test_df["MLM CLint pred_ensemble_mean"].values.tolist() 

hlm_metrics_dict = compute_metrics(hlm, hlm_ens_pred, target="HLM CLint")
mlm_metrics_dict = compute_metrics(mlm, mlm_ens_pred, target = "MLM CLint")

hlm_metrics_df = pd.DataFrame([hlm_metrics_dict])
mlm_metrics_df = pd.DataFrame([mlm_metrics_dict])

metrics_list = [hlm_metrics_df, mlm_metrics_df]

combined_metrics = pd.concat(metrics_list)

# print(combined_metrics) # 2rowsx6cols, looks good

ens_met_path = os.path.join(SUMMARY_DIR, "ensemble_metrics.csv")
combined_metrics.to_csv(ens_met_path, index=False)
print(f"Wrote {ens_met_path}")