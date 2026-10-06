from pathlib import Path
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parents[1]

# Dataset locations
IMAGE_DIR = ROOT / "data" / "raw" / "ISIC2018_Task3_Training_Input"
CSV_PATH = (
    ROOT
    / "data"
    / "raw"
    / "ISIC2018_Task3_Training_GroundTruth"
    / "ISIC2018_Task3_Training_GroundTruth.csv"
)

# Output
OUTPUT_PATH = ROOT / "data" / "manifest.csv"

# The seven ISIC classes
CLASS_COLUMNS = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC",
]


def main():
    print("Loading ground-truth CSV...")
    df = pd.read_csv(CSV_PATH)

    print(f"CSV rows: {len(df)}")
    print(f"Image directory: {IMAGE_DIR}")

    # Create lookup:
    # ISIC_0034305 -> NV
    label_lookup = {}

    for _, row in df.iterrows():
        image_id = str(row["image"]).strip()

        # Find the class whose value is 1
        found_class = None

        for class_name in CLASS_COLUMNS:
            if float(row[class_name]) == 1.0:
                found_class = class_name
                break

        if found_class is not None:
            label_lookup[image_id] = found_class

    print(f"Labels loaded: {len(label_lookup)}")

    # Find images
    image_files = sorted(IMAGE_DIR.glob("*.jpg"))

    print(f"Images found: {len(image_files)}")

    records = []
    missing_labels = []

    for image_path in image_files:
        # IMPORTANT:
        # CSV contains ISIC_0034305
        # filename is ISIC_0034305.jpg
        # Therefore we use .stem to remove .jpg
        image_id = image_path.stem

        class_name = label_lookup.get(image_id)

        if class_name is None:
            missing_labels.append(image_id)
            continue

        records.append(
            {
                "image": str(image_path.relative_to(ROOT)),
                "image_id": image_id,
                "label": class_name,
            }
        )

    # Create manifest
    manifest = pd.DataFrame(records)

    manifest.to_csv(OUTPUT_PATH, index=False)

    print()
    print("====================================")
    print("MANIFEST CREATION COMPLETE")
    print("====================================")
    print(f"Manifest: {OUTPUT_PATH}")
    print(f"Rows written: {len(manifest)}")
    print(f"Missing labels: {len(missing_labels)}")

    if missing_labels:
        print()
        print("First missing labels:")
        for image_id in missing_labels[:20]:
            print(f"  WARNING: no class found for {image_id}")

    print()
    print("Class distribution:")
    print(manifest["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()