# scripts/reset_model.py

import torch
from src.model import UNet
from pathlib import Path

# Configuration
MODEL_PATH = Path("results/unet_model.pth")
IN_CHANNELS = 16
OUT_CHANNELS = 1

def reset_model():
    # Create new UNet with random weights
    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model reset: saved fresh weights to {MODEL_PATH}")

if __name__ == "__main__":
    reset_model()

