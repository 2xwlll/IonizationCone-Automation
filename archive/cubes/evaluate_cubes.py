# scripts/forward_test.py

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.model import UNet
from src.ionization_dataset import IonizationDataset
from pathlib import Path

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet(in_channels=16, out_channels=1).to(device)
model.load_state_dict(torch.load("results/unet_model.pth", map_location=device))
model.eval()

# Load dataset
dataset = IonizationDataset(image_dir="data/processed_3d", mask_dir="data/masks")
dataloader = DataLoader(dataset, batch_size=1)

# Create results directory
results_dir = Path("results/forward_vis")
results_dir.mkdir(parents=True, exist_ok=True)

# Generate and save visualizations
for i, (x, y) in enumerate(dataloader):
    if i >= 10:
        break

    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        pred = model(x)

    input_cube = x[0].cpu().numpy()        # (16, 128, 128)
    output_map = pred[0, 0].cpu().numpy()  # (128, 128)
    mask = y[0, 0].cpu().numpy()           # (128, 128)

    # Plot input, prediction, and ground truth
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_cube.mean(axis=0), cmap='inferno')
    axes[0].set_title("Mean Input Cube")
    axes[0].axis('off')

    axes[1].imshow(output_map, cmap='viridis')
    axes[1].set_title("Model Prediction")
    axes[1].axis('off')

    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis('off')

    plt.tight_layout()
    save_path = results_dir / f"prediction_{i:03}.png"
    plt.savefig(save_path)
    plt.close()

print("✅ Visualizations saved to results/forward_vis")

