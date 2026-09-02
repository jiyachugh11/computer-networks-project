import os
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = "data/processed"
SPLITS_DIR = "data/splits"

rows = []

# Find every PNG and use its parent folder as the label
for label in os.listdir(PROCESSED_DIR):
    label_dir = os.path.join(PROCESSED_DIR, label)

    if not os.path.isdir(label_dir):
        continue

    for filename in os.listdir(label_dir):
        if filename.lower().endswith(".png"):
            image_path = os.path.join(label_dir, filename)
            rows.append((image_path, label))

df = pd.DataFrame(rows, columns=["image_path", "label"])

print("Total images:", len(df))
print("\nImages per class:")
print(df["label"].value_counts())

# 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# Split remaining 30% into 15% validation and 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

os.makedirs(SPLITS_DIR, exist_ok=True)

train_df.to_csv(
    os.path.join(SPLITS_DIR, "train.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(SPLITS_DIR, "val.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(SPLITS_DIR, "test.csv"),
    index=False
)

print("\nCreated:")
print("train.csv:", len(train_df))
print("val.csv:", len(val_df))
print("test.csv:", len(test_df))