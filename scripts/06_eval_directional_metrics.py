import numpy as np
import pandas as pd

SMILES = "SMILES"
PAPP = "Caco-2 Permeability Papp A>B"
EFFLUX = "Caco-2 Permeability Efflux"

TRUE_VAL = "data/processed/caco2_val_paired_temporal.csv"

# pick one seed for intuition (can loop later)
SINGLE_PAPP = "results/chemprop_single/papp/seed_0/model_0/val_predictions.csv"
SINGLE_EFFLUX = "results/chemprop_single/efflux/seed_0/model_0/val_predictions.csv"
MULTI = "results/chemprop_multi/seed_0/model_0/val_predictions.csv"

# Decision thresholds (edit these to whatever feels meaningful)
PAPP_GOOD = 10.0       # example: permeability "good" if >= 10
EFFLUX_GOOD = 2.0      # example: efflux "good" if <= 2

TOP_FRAC = 0.10        # top 10% enrichment


def spearman(x, y):
    # Spearman = Pearson on ranks
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def top_frac_overlap(y_true, y_pred, top_frac, higher_is_better=True):
    n = len(y_true)
    k = max(1, int(round(n * top_frac)))

    if higher_is_better:
        top_true = set(np.argsort(-y_true)[:k])
        top_pred = set(np.argsort(-y_pred)[:k])
    else:
        top_true = set(np.argsort(y_true)[:k])
        top_pred = set(np.argsort(y_pred)[:k])

    return float(len(top_true & top_pred) / k)


def threshold_accuracy(y_true, y_pred, thr, higher_is_better=True):
    if higher_is_better:
        true_good = (y_true >= thr)
        pred_good = (y_pred >= thr)
    else:
        true_good = (y_true <= thr)
        pred_good = (y_pred <= thr)

    return float(np.mean(true_good == pred_good))


def load_pred(path, col):
    df = pd.read_csv(path)[[SMILES, col]].copy()
    return df


def main():
    true = pd.read_csv(TRUE_VAL)[[SMILES, PAPP, EFFLUX]].copy()

    sp = load_pred(SINGLE_PAPP, PAPP)
    se = load_pred(SINGLE_EFFLUX, EFFLUX)
    m = pd.read_csv(MULTI)[[SMILES, PAPP, EFFLUX]].copy()

    # merge
    tp = true.merge(sp, on=SMILES, suffixes=("_true", "_pred"))
    te = true.merge(se, on=SMILES, suffixes=("_true", "_pred"))
    mp = true.merge(m[[SMILES, PAPP]], on=SMILES, suffixes=("_true", "_pred"))
    me = true.merge(m[[SMILES, EFFLUX]], on=SMILES, suffixes=("_true", "_pred"))

    # Papp: higher is better
    print("Papp (higher is better)")
    print("  Spearman single:", spearman(tp[PAPP + "_true"], tp[PAPP + "_pred"]))
    print("  Spearman multi :", spearman(mp[PAPP + "_true"], mp[PAPP + "_pred"]))
    print("  Top% overlap single:", top_frac_overlap(tp[PAPP + "_true"].to_numpy(), tp[PAPP + "_pred"].to_numpy(), TOP_FRAC, True))
    print("  Top% overlap multi :", top_frac_overlap(mp[PAPP + "_true"].to_numpy(), mp[PAPP + "_pred"].to_numpy(), TOP_FRAC, True))
    print("  Threshold acc single:", threshold_accuracy(tp[PAPP + "_true"].to_numpy(), tp[PAPP + "_pred"].to_numpy(), PAPP_GOOD, True))
    print("  Threshold acc multi :", threshold_accuracy(mp[PAPP + "_true"].to_numpy(), mp[PAPP + "_pred"].to_numpy(), PAPP_GOOD, True))

    # Efflux: lower is better
    print("\nEfflux (lower is better)")
    print("  Spearman single:", spearman(te[EFFLUX + "_true"], te[EFFLUX + "_pred"]))
    print("  Spearman multi :", spearman(me[EFFLUX + "_true"], me[EFFLUX + "_pred"]))
    print("  Top% overlap single:", top_frac_overlap(te[EFFLUX + "_true"].to_numpy(), te[EFFLUX + "_pred"].to_numpy(), TOP_FRAC, False))
    print("  Top% overlap multi :", top_frac_overlap(me[EFFLUX + "_true"].to_numpy(), me[EFFLUX + "_pred"].to_numpy(), TOP_FRAC, False))
    print("  Threshold acc single:", threshold_accuracy(te[EFFLUX + "_true"].to_numpy(), te[EFFLUX + "_pred"].to_numpy(), EFFLUX_GOOD, False))
    print("  Threshold acc multi :", threshold_accuracy(me[EFFLUX + "_true"].to_numpy(), me[EFFLUX + "_pred"].to_numpy(), EFFLUX_GOOD, False))


if __name__ == "__main__":
    main()
