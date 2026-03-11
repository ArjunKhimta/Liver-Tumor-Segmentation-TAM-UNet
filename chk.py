import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import LiverDataset
from src.models.tam_unet import TAM_UNet

dataset = LiverDataset(
    images_dir="data/processed/images",
    masks_dir="data/processed/masks"
)

model = TAM_UNet()
model.load_state_dict(torch.load("tam_best_model.pth", map_location="cpu"))
model.eval()

image, mask = dataset[50]

with torch.no_grad():
    pred = model(image.unsqueeze(0))

    if isinstance(pred, tuple):
        pred = pred[0]

pred = torch.argmax(pred, dim=1)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Image")
plt.imshow(image.squeeze(), cmap="gray")

plt.subplot(1,3,2)
plt.title("Ground Truth")
plt.imshow(mask.squeeze())

plt.subplot(1,3,3)
plt.title("Prediction")
plt.imshow(pred.squeeze())

plt.show()