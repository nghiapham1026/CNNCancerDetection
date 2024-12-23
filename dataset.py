import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class HistopathologyDataset(Dataset):
    """
    Custom dataset for histopathologic cancer detection images.
    Each item yields (transformed_image, label).
    """
    def __init__(self, df, img_dir, transform=None):
        """
        df: DataFrame with columns ['id', 'label']
        img_dir: directory containing .tif images
        transform: torchvision transform pipeline
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id, label = row['id'], row['label']
        img_path = os.path.join(self.img_dir, img_id + ".tif")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
