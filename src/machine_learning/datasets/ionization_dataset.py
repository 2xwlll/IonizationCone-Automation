import os
import numpy as np
import torch
from torch.utils.data import Dataset

class IonizationConeDataset2D(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, normalize=True):
        """
        Args:
            image_dir (str): Directory containing .npy images
            mask_dir (str): Directory containing .npy masks
            transform (callable, optional): joint transform (image+mask)
            normalize (bool): apply per-image normalization
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.normalize = normalize

        # --- Collect files ---
        image_files = [f for f in os.listdir(image_dir) if f.endswith(".npy")]
        mask_files  = [f for f in os.listdir(mask_dir) if f.endswith(".npy")]

        # --- Build lookup by base name ---
        image_map = {os.path.splitext(f)[0]: f for f in image_files}
        mask_map  = {os.path.splitext(f)[0]: f for f in mask_files}

        # --- Find intersection ---
        common_keys = sorted(set(image_map.keys()) & set(mask_map.keys()))

        if len(common_keys) == 0:
            raise ValueError("No matching image/mask pairs found.")

        self.pairs = [
            (image_map[k], mask_map[k]) for k in common_keys
        ]

        print(f"[Dataset] Loaded {len(self.pairs)} matched samples.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_file, mask_file = self.pairs[idx]

        img_path  = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        image = np.load(img_path).astype(np.float32)
        mask  = np.load(mask_path).astype(np.float32)

        # A spectral sample is stored channel-first as (wavelength, y, x),
        # while its segmentation target remains a single (y, x) mask.
        if image.ndim not in (2, 3):
            raise ValueError(
                f"Expected 2D image or channel-first spectral cube, got "
                f"{image.shape} in {img_file}"
            )
        if image.shape[-2:] != mask.shape:
            raise ValueError(
                f"Spatial shape mismatch: image={image.shape}, mask={mask.shape}"
            )

        # --- Normalize image ---
        if self.normalize:
            std = image.std()
            if std > 0:
                image = (image - image.mean()) / (std + 1e-6)
            else:
                image = image - image.mean()

        # --- Ensure mask is binary ---
        mask = (mask > 0.5).astype(np.float32)

        # Grayscale images need a channel; cubes already use wavelength as it.
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        mask  = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(image)
        mask  = torch.from_numpy(mask)

        # --- Joint transform ---
        if self.transform:
            stacked = torch.cat([image, mask], dim=0)
            stacked = self.transform(stacked)
            image, mask = stacked[0:1], stacked[1:2]

        return image, mask
