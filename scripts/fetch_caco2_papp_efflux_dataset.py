from datasets import load_dataset

DATASET_NAME = "openadmet/openadmet-expansionrx-challenge-data"

# Column names as they appear in the ML-ready dataset
SMILES_COL = "SMILES"
TARGETS = [
    "Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux",
]

OUT_TRAIN = "data/raw/caco2_train.csv"
OUT_TEST  = "data/raw/caco2_test.csv"

def main():
    cols = [SMILES_COL] + TARGETS

    train = load_dataset(DATASET_NAME, split="train").to_pandas()
    test  = load_dataset(DATASET_NAME, split="test").to_pandas()

    # Keep only the columns we need
    train = train[cols]
    test = test[cols]

    train.to_csv(OUT_TRAIN, index=False)
    test.to_csv(OUT_TEST, index=False)

    print("Wrote:")
    print(f" - {OUT_TRAIN}: {train.shape}")
    print(f" - {OUT_TEST}:  {test.shape}")
    print("\nMissing labels in TRAIN:")
    print(train[TARGETS].isna().sum().to_string())
    print("\nMissing labels in TEST:")
    print(test[TARGETS].isna().sum().to_string())

if __name__ == "__main__":
    main()
