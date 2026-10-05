from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"
VAL_CSV = ROOT / "data" / "splits" / "val.csv"

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 1

NUM_WORKERS = 0

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
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ============================================================
# DATA
# ============================================================

print("Loading datasets...")

train_dataset = ISICDataset(
    TRAIN_CSV,
    transform=train_transform
)

val_dataset = ISICDataset(
    VAL_CSV,
    transform=val_transform
)

print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# MODEL
# ============================================================

print("Creating model...")

model = models.resnet18(weights="DEFAULT")

# Replace final layer for 7 ISIC classes
model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

model = model.to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("Calculating class weights...")

train_labels = [
    CLASS_TO_INDEX[label]
    for label in train_dataset.df["label"]
]

class_counts = torch.bincount(
    torch.tensor(train_labels),
    minlength=len(CLASS_NAMES)
).float()

class_weights = len(train_labels) / (
    len(CLASS_NAMES) * class_counts
)

class_weights = class_weights.to(DEVICE)

print("Class counts:")
for name, count in zip(CLASS_NAMES, class_counts):
    print(f"  {name}: {int(count)}")

print("Class weights:")
for name, weight in zip(CLASS_NAMES, class_weights):
    print(f"  {name}: {weight.item():.4f}")


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


# ============================================================
# TRAINING
# ============================================================

print()
print("====================================")
print("STARTING TRAINING")
print("====================================")

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if (batch_index + 1) % 50 == 0:

            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_index + 1}/{len(train_loader)} | "
                f"Loss {loss.item():.4f}"
            )

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss_total = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_total += loss.item()

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_loss_total / len(val_loader)
    val_accuracy = val_correct / val_total

    print()
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train loss:     {train_loss:.4f}"
    )

    print(
        f"Train accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Val loss:       {val_loss:.4f}"
    )

    print(
        f"Val accuracy:   {val_accuracy:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "isic_resnet18_test.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
    },
    model_path,
)

print()
print("====================================")
print("TRAINING COMPLETE")
print("====================================")
print(f"Model saved to:")
print(model_path)