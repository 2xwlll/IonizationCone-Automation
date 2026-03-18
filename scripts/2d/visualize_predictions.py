# scripts/2d/visualize_predictions.py

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.model_2d import UNet
from src.datasets.ionization_dataset_2d import Ionization2DDataset

# ===== CONFIG =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR = "data/2d/processed"
MASK_DIR = "data/2d/masks"
MODEL_PATH = "results/unet_model_2d.pth"
SAVE_DIR = Path("results/visuals")
BATCH_SIZE = 1
NUM_TO_VIS = 5

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ===== Load Data =====
dataset = Ionization2DDataset(IMG_DIR, MASK_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== Load Model =====
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== Inference + Plotting =====
print("🔍 Visualizing predictions...")
with torch.no_grad():
    for i, (inputs, targets) in enumerate(loader):
        if i >= NUM_TO_VIS:
            break

        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs)

        input_np = inputs[0][0].cpu().numpy()
        target_np = targets[0][0].cpu().numpy()
        pred_np = probs[0][0].cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(input_np, cmap="gray")
        axs[0].set_title("Input Image")

        axs[1].imshow(target_np, cmap="plasma")
        axs[1].set_title("Ground Truth Mask")

        axs[2].imshow(pred_np, cmap="viridis")
        axs[2].set_title("Model Prediction")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        save_path = SAVE_DIR / f"prediction_{i:03}.png"
        plt.savefig(save_path)
        plt.show()

        print(f"✅ Saved prediction image to {save_path}")

