import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TrafficDataset(Dataset):
    def __init__(self, csv_file, class_to_idx=None, image_size=28):
        self.data = pd.read_csv(csv_file)

        if class_to_idx is None:
            classes = sorted(self.data["label"].unique())
            self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image = Image.open(row["image_path"]).convert("L")
        image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "class_name": row["label"]
        }


if __name__ == "__main__":
    import tempfile
    from PIL import Image as PILImage
    import numpy as np

    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, "dummy.csv")
    rows = []
    for i, label in enumerate(["Skype", "Zoom"] * 2):
        img_path = os.path.join(tmp_dir, f"img_{i}.png")
        arr = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        PILImage.fromarray(arr, mode="L").save(img_path)
        rows.append((img_path, label))

    pd.DataFrame(rows, columns=["image_path", "label"]).to_csv(csv_path, index=False)

    dataset = TrafficDataset(csv_path)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}, Label: {sample['label']}, Class: {sample['class_name']}")