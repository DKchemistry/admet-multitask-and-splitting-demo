import os
import glob
import pathlib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau

RESULTS_CSVS = "results/hlm_mlm_temporal_80_20_test/**/predictions.csv"
TEST = "data/processed/hlm_mlm_paired_log10_test.csv" 
SUMMARY_DIR = "results/hlm_mlm_temporal_80_20_test/summary"

list_of_dfs = []

def collect_dataframes():
    for filename in glob.glob(RESULTS_CSVS, recursive=True):
        path = pathlib.Path(filename)
        # print(path)
        seed = path.parent.name
        # print(seed)
        task_family = path.parent.parent.name
        # print(task_family)
        model_type = path.parent.parent.parent.name
        # print(model_type)
        rename_HLM = "HLM CLint " + model_type + "_" + task_family  + "_" + seed
        rename_MLM = "MLM CLint " + model_type + "_" + task_family  + "_" + seed

        # print(rename_HLM)
        # print(rename_MLM)

        df = pd.read_csv(path)
        df.rename(
          {
            "HLM CLint":rename_HLM,
            "MLM CLint":rename_MLM},
            axis = "columns", inplace = True
            )
        
        if task_family == "hlm":
          df.drop(labels=rename_MLM, axis=1, inplace=True)
        elif task_family == "mlm":
          df.drop(labels=rename_HLM, axis=1, inplace=True)
        elif task_family == "hlm_mlm":
          pass

        list_of_dfs.append(df)

def main():
    collect_dataframes()

    test_df = pd.read_csv(TEST)

    for n in range(len(list_of_dfs)):
        test_df = pd.merge(left=test_df, right=list_of_dfs[n], on="SMILES")

    group_dict = {}
    for col in test_df.columns:
        # col is a string
        if "_seed_" in col:
            # remove the suffix to make a key
            split_col = col.split("_seed_")
            keys = split_col[0]
            # print(keys)
            # have to init the key
            # before addings vals
            if keys not in group_dict:
                group_dict[keys] = []
            group_dict[keys].append(col)

    # get ensemble means/sdev
    for key in group_dict:
        # print(key)
        subset_cols = group_dict[key]
        col_mean_name = key + "_mean"
        col_std_name = key + "_sdev"
        test_df[col_mean_name] = test_df[subset_cols].mean(axis=1)
        test_df[col_std_name] = test_df[subset_cols].std(axis=1)

    # Save per molecule predictions for seed and ensemble
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    per_molecule_prediction_path = os.path.join(SUMMARY_DIR, "per_molecule_predictions.csv")
    test_df.to_csv(per_molecule_prediction_path, index=False)
    print(f"Wrote molecule level predictions to {per_molecule_prediction_path}")

    ensemble_metrics_dfs = []
    for col in test_df.columns:
        if col.startswith("HLM CLint") and col.endswith("mean"):
            hlm_truth_col_list = test_df["HLM CLint"].values.tolist()
            hlm_pred_col_list = test_df[col].values.tolist()
            rmse = mean_squared_error(hlm_truth_col_list, hlm_pred_col_list) ** 0.5
            mae = mean_absolute_error(hlm_truth_col_list, hlm_pred_col_list)
            r2 = r2_score(hlm_truth_col_list, hlm_pred_col_list)
            tau, tau_pvalue = kendalltau(hlm_truth_col_list, hlm_pred_col_list)

            model_family_name = col.split(" ")[2].removesuffix("_mean")

            hlm_metrics_df = pd.DataFrame(
                [
                    {
                        "target": "HLM CLint",
                        "model_family": model_family_name,
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2,
                        "kendall_tau": tau,
                        "kendall_tau_pvalue": tau_pvalue,
                    }
                ]
            )
            ensemble_metrics_dfs.append(hlm_metrics_df)

        if col.startswith("MLM CLint") and col.endswith("mean"):
            mlm_truth_col_list = test_df["MLM CLint"].values.tolist()
            mlm_pred_col_list = test_df[col].values.tolist()
            rmse = mean_squared_error(mlm_truth_col_list, mlm_pred_col_list) ** 0.5
            mae = mean_absolute_error(mlm_truth_col_list, mlm_pred_col_list)
            r2 = r2_score(mlm_truth_col_list, mlm_pred_col_list)
            tau, tau_pvalue = kendalltau(mlm_truth_col_list, mlm_pred_col_list)

            model_family_name = col.split(" ")[2].removesuffix("_mean")

            mlm_metrics_df = pd.DataFrame(
                [
                    {
                        "target": "MLM CLint",
                        "model_family": model_family_name,
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2,
                        "kendall_tau": tau,
                        "kendall_tau_pvalue": tau_pvalue,
                    }
                ]
            )
            ensemble_metrics_dfs.append(mlm_metrics_df)

    ensemble_metrics_df = pd.concat(ensemble_metrics_dfs)

    # Save ensemble metrics
    ensemble_summary_path = os.path.join(SUMMARY_DIR, "ensemble_metrics.csv")
    ensemble_metrics_df.to_csv(ensemble_summary_path, index=False)
    print(f"Wrote ensemble metrics to {ensemble_summary_path}")

if __name__ == "__main__":
    main()
