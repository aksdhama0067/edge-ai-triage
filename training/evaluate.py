from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

VAL_CSV = ROOT / "data" / "splits" / "val.csv"

MODEL_PATH = ROOT / "models" / "isic_resnet18_test.pth"

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "isic_baseline_evaluation.txt"


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "AKIEC",
    "BCC",
    "BKL",
    "DF",
    "MEL",
    "NV",
    "VASC",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# DATASET
# ============================================================

class ISICDataset(Dataset):

    def __init__(self, csv_path, transform=None):

        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = ROOT / row["image"]

        image = Image.open(image_path).convert("RGB")

        label = CLASS_TO_INDEX[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORM
# ============================================================

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("Loading validation dataset...")

val_dataset = ISICDataset(
    VAL_CSV,
    transform=val_transform
)

print(f"Validation images: {len(val_dataset)}")

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# RUN EVALUATION
# ============================================================

print()
print("====================================")
print("STARTING EVALUATION")
print("====================================")

all_labels = []
all_predictions = []

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(val_loader):

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

        if (batch_index + 1) % 50 == 0:
            print(
                f"Evaluated batch "
                f"{batch_index + 1}/{len(val_loader)}"
            )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0,
)


report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0,
)

matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES))),
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("====================================")
print("EVALUATION RESULTS")
print("====================================")

print(f"Accuracy:          {accuracy:.4f}")
print(f"Macro Precision:   {macro_precision:.4f}")
print(f"Macro Recall:      {macro_recall:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")
print(f"Weighted F1:       {weighted_f1:.4f}")

print()
print("Classification report:")
print(report)

print("Confusion matrix:")
print(matrix)


# ============================================================
# SAVE REPORT
# ============================================================

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("ISIC 2018 BASELINE MODEL EVALUATION\n")
    f.write("====================================\n\n")

    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Validation images: {len(val_dataset)}\n\n")

    f.write("Overall metrics\n")
    f.write("----------------\n")
    f.write(f"Accuracy:        {accuracy:.4f}\n")
    f.write(f"Macro Precision: {macro_precision:.4f}\n")
    f.write(f"Macro Recall:    {macro_recall:.4f}\n")
    f.write(f"Macro F1:        {macro_f1:.4f}\n")
    f.write(f"Weighted F1:     {weighted_f1:.4f}\n\n")

    f.write("Classification report\n")
    f.write("--------------------\n")
    f.write(report)

    f.write("\nConfusion matrix\n")
    f.write("----------------\n")

    f.write(
        "Rows = actual class\n"
        "Columns = predicted class\n\n"
    )

    f.write("              ")
    for name in CLASS_NAMES:
        f.write(f"{name:>8}")
    f.write("\n")

    for i, row in enumerate(matrix):

        f.write(f"{CLASS_NAMES[i]:>8} ")

        for value in row:
            f.write(f"{value:>8}")

        f.write("\n")


print()
print("====================================")
print("EVALUATION COMPLETE")
print("====================================")

print(f"Report saved to:")
print(REPORT_PATH)