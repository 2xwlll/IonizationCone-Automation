from torch.utils.data import DataLoader
from src.ionization_dataset import IonizationDataset

dataset = IonizationDataset(mode='synthetic', use_masks=True)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    images, masks = batch
    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape: {masks.shape}")
    break

