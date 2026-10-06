from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# Project root
ROOT = Path(__file__).resolve().parents[1]

# Input
MANIFEST_PATH = ROOT / "data" / "manifest.csv"

# Output
SPLIT_DIR = ROOT / "data" / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = SPLIT_DIR / "train.csv"
VAL_PATH = SPLIT_DIR / "val.csv"


def main():
    print("Loading manifest...")

    df = pd.read_csv(MANIFEST_PATH)

    print(f"Total images: {len(df)}")

    # Make sure every row has a valid label
    if df["label"].isna().any():
        raise ValueError("Some images have missing labels.")

    # Stratified 80/20 split
    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    # Save
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)

    print()
    print("====================================")
    print("TRAIN / VALIDATION SPLIT COMPLETE")
    print("====================================")

    print(f"Training images:   {len(train_df)}")
    print(f"Validation images: {len(val_df)}")

    print()
    print("Training distribution:")
    print(train_df["label"].value_counts().sort_index())

    print()
    print("Validation distribution:")
    print(val_df["label"].value_counts().sort_index())

    print()
    print(f"Train CSV: {TRAIN_PATH}")
    print(f"Val CSV:   {VAL_PATH}")


if __name__ == "__main__":
    main()