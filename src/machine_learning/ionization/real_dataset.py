# src/ionization/real_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class RealJWSTDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = sorted(Path(image_dir).glob("*.npy"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        cube = np.load(self.image_paths[idx])  # shape: (16, 128, 128)
        cube = torch.from_numpy(cube).float()  # no unsqueeze
        return cube, self.image_paths[idx].name

