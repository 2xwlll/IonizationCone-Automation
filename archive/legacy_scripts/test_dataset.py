import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.ionization_dataset import IonizationDataset

dataset = IonizationDataset()

print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Sample shape: {sample.shape}")
print(f"Dtype: {sample.dtype}")

