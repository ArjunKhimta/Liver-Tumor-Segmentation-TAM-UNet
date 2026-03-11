import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from src.dataset.liver_dataset import LiverDataset
from src.models.unet_baseline import UNetBaseline

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Load validation dataset
val_dataset = LiverDataset("data/interim/val")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# Load model
model = UNetBaseline().to(device)
model.load_state_dict(torch.load("models/unet_baseline_best.pth", map_location=device))
model.eval()

print("Model loaded successfully.")

# Predict on a few slices
num_images_to_show = 5

with torch.no_grad():
    for i, (image, mask) in enumerate(val_loader):
        image = image.to(device)

        output = model(image)
        output = torch.sigmoid(output)

        # Convert to binary mask
        pred_mask = (output > 0.5).float()

        image_np = image.cpu().squeeze().numpy()
        mask_np = mask.squeeze().numpy()
        pred_np = pred_mask.cpu().squeeze().numpy()

        # Plot
        plt.figure(figsize=(12, 4))

        # Original image
        plt.subplot(1, 3, 1)
        plt.title("Original CT")
        plt.imshow(image_np, cmap="gray")
        plt.axis("off")

        # Ground truth
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(mask_np, cmap="gray")
        plt.axis("off")

        # Prediction overlay
        plt.subplot(1, 3, 3)
        plt.title("Prediction Overlay")
        plt.imshow(image_np, cmap="gray")
        plt.imshow(pred_np, cmap="jet", alpha=0.5)
        plt.axis("off")

        plt.show()

        if i + 1 >= num_images_to_show:
            break