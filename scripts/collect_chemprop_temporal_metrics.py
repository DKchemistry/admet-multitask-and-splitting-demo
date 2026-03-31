import os
import glob
import pathlib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import bootstrap, kendalltau
import numpy as np

RESULTS_CSVS = "results/hlm_mlm_temporal_80_20_test/**/predictions.csv"
TEST = "data/processed/hlm_mlm_paired_log10_test.csv" 
SUMMARY_DIR = "results/hlm_mlm_temporal_80_20_test/summary"

list_of_dfs = []


def mae_statistic(errors):
    return np.mean(errors)


def rmse_statistic(errors):
    return np.sqrt(np.mean(errors))


def r2_statistic(y_true, y_pred):
    return r2_score(y_true, y_pred)


def kendall_tau_statistic(y_true, y_pred):
    return kendalltau(y_true, y_pred).statistic


def compute_metrics(y_true, y_pred, target):
    mae = mean_absolute_error(y_true, y_pred)
    abs_errors = np.abs(y_true - y_pred)
    mae_bootstrap = bootstrap(
        (abs_errors,), mae_statistic, n_resamples=1000, confidence_level=0.95
    )

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    sq_errors = abs_errors**2
    rmse_bootstrap = bootstrap(
        (sq_errors,), rmse_statistic, n_resamples=1000, confidence_level=0.95
    )

    r2 = r2_score(y_true, y_pred)
    r2_bootstrap = bootstrap(
        (y_true, y_pred),
        r2_statistic,
        n_resamples=1000,
        paired=True,
        confidence_level=0.95,
    )

    tau, tau_pvalue = kendalltau(y_true, y_pred)
    tau_bootstrap = bootstrap(
        (y_true, y_pred),
        kendall_tau_statistic,
        n_resamples=1000,
        paired=True,
        confidence_level=0.95,
    )

    return {
        "target": target,
        "mae": mae,
        "mae_low_ci": mae_bootstrap.confidence_interval.low,
        "mae_high_ci": mae_bootstrap.confidence_interval.high,
        "rmse": rmse,
        "rmse_low_ci": rmse_bootstrap.confidence_interval.low,
        "rmse_high_ci": rmse_bootstrap.confidence_interval.high,
        "r2": r2,
        "r2_low_ci": r2_bootstrap.confidence_interval.low,
        "r2_high_ci": r2_bootstrap.confidence_interval.high,
        "kendall_tau": tau,
        "kendall_tau_low_ci": tau_bootstrap.confidence_interval.low,
        "kendall_tau_high_ci": tau_bootstrap.confidence_interval.high,
        "kendall_tau_pvalue": tau_pvalue,
    }


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
            hlm_truth_col_list = test_df["HLM CLint"].to_numpy()
            hlm_pred_col_list = test_df[col].to_numpy()

            hlm_metrics_dict = compute_metrics(hlm_truth_col_list, hlm_pred_col_list, target="HLM CLint")
            hlm_metrics_df = pd.DataFrame([hlm_metrics_dict])

            model_family_name = col.split(" ")[2].removesuffix("_mean")
            hlm_metrics_df["model_family"] = model_family_name

            ensemble_metrics_dfs.append(hlm_metrics_df)

        if col.startswith("MLM CLint") and col.endswith("mean"):
            mlm_truth_col_list = test_df["MLM CLint"].to_numpy()
            mlm_pred_col_list = test_df[col].to_numpy()

            mlm_metrics_dict = compute_metrics(mlm_truth_col_list, mlm_pred_col_list, target = "MLM CLint")
            mlm_metrics_df = pd.DataFrame([mlm_metrics_dict])

            model_family_name = col.split(" ")[2].removesuffix("_mean")
            mlm_metrics_df["model_family"] = model_family_name

            ensemble_metrics_dfs.append(mlm_metrics_df)

    ensemble_metrics_df = pd.concat(ensemble_metrics_dfs)

    # Save ensemble metrics
    ensemble_summary_path = os.path.join(SUMMARY_DIR, "ensemble_metrics.csv")
    ensemble_metrics_df.to_csv(ensemble_summary_path, index=False)
    print(f"Wrote ensemble metrics to {ensemble_summary_path}")

if __name__ == "__main__":
    main()
