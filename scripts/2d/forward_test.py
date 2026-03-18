
# scripts/2d/forward_test_2d.py

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.datasets import ionization_dataset_2d
from src.models.model_2d import UNet
from src.datasets.ionization_dataset_2d import IonizationDataset2D

# ===== CONFIG =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR = "data/2d/processed"
MASK_DIR = "data/2d/masks"
BATCH_SIZE = 2
MODEL_PATH = "results/unet_best_2d.pth"

# ===== Load dataset =====
print("Loading dataset...")
dataset = IonizationDataset2D (IMG_DIR, MASK_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== Load model =====
print("Loading model...")
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== Run forward pass =====
print("Running forward pass...")
inputs, masks = next(iter(loader))
inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)

with torch.no_grad():
    outputs = model(inputs)
    probs = torch.sigmoid(outputs)  # Apply sigmoid to see probability

print("Forward pass complete.")
print(f" - Input shape:  {inputs.shape}")
print(f" - Output shape: {outputs.shape}")
print(f" - Target shape: {masks.shape}")

# ===== Visualize first few results =====
for i in range(min(BATCH_SIZE, 3)):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(inputs[i][0].cpu(), cmap="gray")
    axs[0].set_title("Input Image")

    axs[1].imshow(masks[i][0].cpu(), cmap="plasma")
    axs[1].set_title("Ground Truth Mask")

    axs[2].imshow(probs[i][0].cpu(), cmap="viridis")
    axs[2].set_title("Model Prediction (sigmoid)")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

