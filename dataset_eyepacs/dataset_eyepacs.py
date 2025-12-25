import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO

class EyePACSDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        # Get all parquet files
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
        # Read all files and concatenate
        dfs = [pd.read_parquet(f) for f in files]
        self.data = pd.concat(dfs, ignore_index=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Extract bytes from dict if needed
        img_dict = row["image"]
        if isinstance(img_dict, dict) and "bytes" in img_dict:
            img_bytes = img_dict["bytes"]
        elif isinstance(img_dict, bytes):
            img_bytes = img_dict
        else:
            raise TypeError(f"Unsupported image format: {type(img_dict)}")
        
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        label = row["label_code"]
        if self.transform:
            img = self.transform(img)
        return img, label
