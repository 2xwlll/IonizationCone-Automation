import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

# Your project imports
from src.datasets.ionization_dataset_2d import IonizationDataset2D
from src.models.model_2d import UNet
from src.losses.combined_BCE_Dice import BCEDiceLoss

# --- Settings ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "results/unet_best_2d.pth"
data_dir = "data/2d/processed/synthetic"  # or "real"
batch_size = 4
save_dir = "results/evaluation_slides"
os.makedirs(save_dir, exist_ok=True)

# --- Load model ---
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Load dataset ---
dataset = IonizationDataset2D(data_dir)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# --- Metrics storage ---
dice_scores = []
precisions = []
recalls = []
ious = []

def dice_coeff(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2 * intersection + eps) / (pred.sum() + target.sum() + eps)

# --- Evaluate ---
examples = []  # store some images for visualization
with torch.no_grad():
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        pred = torch.sigmoid(model(x)) > 0.5  # binary mask
        for b in range(x.size(0)):
            dice = dice_coeff(pred[b], y[b]).item()
            dice_scores.append(dice)
            precisions.append(precision_score(y[b].cpu().numpy().flatten(),
                                              pred[b].cpu().numpy().flatten()))
            recalls.append(recall_score(y[b].cpu().numpy().flatten(),
                                        pred[b].cpu().numpy().flatten()))
            ious.append(jaccard_score(y[b].cpu().numpy().flatten(),
                                      pred[b].cpu().numpy().flatten()))
            # save some examples for slides
            if len(examples) < 5:
                examples.append((x[b].cpu().numpy()[0],  # assuming single-channel
                                 y[b].cpu().numpy()[0],
                                 pred[b].cpu().numpy()[0]))

# --- Summary stats ---
print("Mean Dice:", np.mean(dice_scores))
print("Mean Precision:", np.mean(precisions))
print("Mean Recall:", np.mean(recalls))
print("Mean IoU:", np.mean(ious))

# --- Plots ---
# Dice histogram
plt.figure(figsize=(6,4))
plt.hist(dice_scores, bins=20, color='skyblue', edgecolor='black')
plt.title("Dice Score Distribution")
plt.xlabel("Dice Score")
plt.ylabel("Number of images")
plt.savefig(os.path.join(save_dir, "dice_histogram.png"))

# Examples for slides
for idx, (img, gt, pred) in enumerate(examples):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(gt, cmap='Reds')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(img, cmap='gray')
    plt.imshow(pred, cmap='Blues', alpha=0.5)
    plt.title("Prediction Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"example_{idx}.png"))
