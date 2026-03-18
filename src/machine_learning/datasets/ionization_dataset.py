
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class IonizationConeDataset2D(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Directory containing .npy images
            mask_dir (str): Directory containing .npy segmentation masks
            transform (callable, optional): Optional transform to apply to both image and mask
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get shared base filenames without extension
        self.image_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith('.npy')
        ])
        self.mask_files = sorted([
            f for f in os.listdir(mask_dir) if f.endswith('.npy')
        ])

        # Match base filenames
        self.image_files = [
            f for f in self.image_files if f in self.mask_files
        ]
        assert len(self.image_files) > 0, "No matching .npy files found between image_dir and mask_dir"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])  # assumes same filename

        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        # Normalize image: zero mean, unit std
        image = (image - image.mean()) / (image.std() + 1e-6)

        # Add channel dim: (H, W) → (1, H, W)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            # Combine into one tensor for joint transforms
            stacked = torch.from_numpy(np.concatenate([image, mask], axis=0))
            stacked = self.transform(stacked)
            image, mask = stacked[0:1], stacked[1:2]
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)

        return image, mask

