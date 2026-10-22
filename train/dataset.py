import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class NucleiDataset(Dataset):
    def __init__(self, split_file, split="train", processed_dir="data/processed"):
        with open(split_file) as f:
            self.ids = json.load(f)[split]
        self.img_dir = Path(processed_dir) / "images"
        self.mask_dir = Path(processed_dir) / "masks"
        self.weight_dir = Path(processed_dir) / "weights"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        iid = self.ids[idx]
        img = np.array(Image.open(self.img_dir / f"{iid}.png")) / 255.0
        mask = np.array(Image.open(self.mask_dir / f"{iid}.png")) > 127
        weight = np.load(self.weight_dir / f"{iid}.npy")

        img = torch.from_numpy(img).float().unsqueeze(0)          # 1 x H x W
        mask = torch.from_numpy(mask.astype(np.int64))            # H x W
        weight = torch.from_numpy(weight).float()                 # H x W
        return img, mask, weight
