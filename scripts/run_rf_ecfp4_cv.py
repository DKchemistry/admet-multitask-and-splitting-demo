import os
import argparse
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau

RDLogger.DisableLog("rdApp.warning")

def ecfp4_from_smiles(smiles_list, n_bits=2048):
    fps = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        fps.append(list(fp))

    return fps


def fit_and_predict_one_target(train_df, val_df, smiles_col, target_col, n_bits, seed):
    x_train = ecfp4_from_smiles(train_df[smiles_col].tolist(), n_bits=n_bits)
    x_val = ecfp4_from_smiles(val_df[smiles_col].tolist(), n_bits=n_bits)

    y_train = train_df[target_col].tolist()
    y_val = val_df[target_col].tolist()

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_val)

    rmse = mean_squared_error(y_val, preds) ** 0.5
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    tau, tau_pvalue = kendalltau(y_val, preds)

    pred_df = val_df[[smiles_col, target_col, "row_id"]].copy()
    pred_df["prediction"] = preds

    metrics_df = pd.DataFrame(
        [
            {
                "target": target_col,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "kendall_tau": tau,
                "kendall_tau_pvalue": tau_pvalue,
            }
        ]
    )

    return pred_df, metrics_df


def run_one_fold_csv(
    data_csv,
    out_root,
    smiles_col,
    target_cols,
    n_bits,
    seed,
):
    df = pd.read_csv(data_csv)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    os.makedirs(out_root, exist_ok=True)

    all_metrics = []

    for target_col in target_cols:
        target_out_dir = os.path.join(
            out_root,
            target_col.replace(" ", "_").replace("/", "_"),
        )
        os.makedirs(target_out_dir, exist_ok=True)

        pred_df, metrics_df = fit_and_predict_one_target(
            train_df,
            val_df,
            smiles_col=smiles_col,
            target_col=target_col,
            n_bits=n_bits,
            seed=seed,
        )

        pred_path = os.path.join(target_out_dir, "predictions.csv")
        metrics_path = os.path.join(target_out_dir, "metrics.csv")

        pred_df.to_csv(pred_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)

        all_metrics.append(metrics_df)

        print(f"Wrote {pred_path}")
        print(f"Wrote {metrics_path}")

    all_metrics_df = pd.concat(all_metrics, axis=0, ignore_index=True)
    all_metrics_path = os.path.join(out_root, "metrics_all_targets.csv")
    all_metrics_df.to_csv(all_metrics_path, index=False)
    print(f"Wrote {all_metrics_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--smiles_col", required=True)
    ap.add_argument("--target_cols", nargs="+", required=True)
    ap.add_argument("--n_folds", type=int, required=True)
    ap.add_argument("--n_repeats", type=int, required=True)
    ap.add_argument("--n_bits", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    for cv_iter in range(args.n_repeats):
        for fold in range(args.n_folds):
            data_csv = os.path.join(
                args.fold_root,
                f"iter_{cv_iter}",
                f"fold_{fold}.csv",
            )

            out_dir = os.path.join(
                args.out_root,
                f"iter_{cv_iter}",
                f"fold_{fold}",
            )

            print("==============================")
            print(f"iter {cv_iter} fold {fold}")
            print(f"data: {data_csv}")
            print(f"out : {out_dir}")
            print("==============================")

            run_one_fold_csv(
                data_csv=data_csv,
                out_root=out_dir,
                smiles_col=args.smiles_col,
                target_cols=args.target_cols,
                n_bits=args.n_bits,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()
