# scripts/2d/evaluate_2d.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

from src.datasets.ionization_dataset_2d import IonizationDataset2D
from src.models.model_2d import UNet

# ===== CONFIG =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "results/unet_model_2d.pth"
IMG_DIR = Path("data/2d/processed")
MASK_DIR = Path("data/2d/masks")
BATCH_SIZE = 1

def evaluate_model():
    print("🔍 Evaluating 2D UNet model...")

    dataset = IonizationDataset2D(data_root=IMG_DIR, mask_root=MASK_DIR, use_masks=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    ious, dices, precisions, recalls = [], [], [], []

    with torch.no_grad():
        for i, (img, mask) in enumerate(loader):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            pred = model(img)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()

            pred_np = pred.cpu().numpy().flatten()
            mask_np = mask.cpu().numpy().flatten()

            ious.append(jaccard_score(mask_np, pred_np))
            dices.append(f1_score(mask_np, pred_np))
            precisions.append(precision_score(mask_np, pred_np))
            recalls.append(recall_score(mask_np, pred_np))

            if i < 3:
                plt.figure(figsize=(10, 3))
                plt.subplot(1, 3, 1)
                plt.imshow(img.cpu().squeeze(), cmap="gray")
                plt.title("Input Image")
                plt.subplot(1, 3, 2)
                plt.imshow(mask.cpu().squeeze(), cmap="gray")
                plt.title("Ground Truth")
                plt.subplot(1, 3, 3)
                plt.imshow(pred.cpu().squeeze(), cmap="plasma")
                plt.title("Prediction")
                plt.tight_layout()
                plt.show()

    print("\n📊 Evaluation Metrics (averaged over dataset):")
    print(f"IoU:       {np.mean(ious):.4f}")
    print(f"Dice:      {np.mean(dices):.4f}")
    print(f"Precision: {np.mean(precisions):.4f}")
    print(f"Recall:    {np.mean(recalls):.4f}")

if __name__ == "__main__":
    evaluate_model()

