import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dataset import LiverDataset
from src.models.unet_baseline import UNetBaseline
from src.models.tam_unet import TAM_UNet


# ==============================
# Device
# ==============================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ==============================
# Overlay Visualization
# ==============================
def overlay_visualization(image, mask, prediction, save_path=None):
    image = image.cpu().numpy().squeeze()
    mask = mask.cpu().numpy()
    prediction = prediction.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Ground Truth Overlay
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(mask, cmap="jet", alpha=0.5, vmin=0, vmax=2)
    axes[1].set_title("Ground Truth Overlay")
    axes[1].axis("off")

    # Prediction Overlay
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(prediction, cmap="jet", alpha=0.5, vmin=0, vmax=2)
    axes[2].set_title("Prediction Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.show()


# ==============================
# Main
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "tam"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=5)

    args = parser.parse_args()

    device = get_device()
    print(f"\nUsing device: {device}")
    print(f"Model: {args.model}")
    print(f"Weights: {args.weights}\n")

    # ==============================
    # Dataset
    # ==============================
    IMAGES_DIR = "data/processed/images"
    MASKS_DIR = "data/processed/masks"

    dataset = LiverDataset(IMAGES_DIR, MASKS_DIR)

    # 🔥 Shuffle to get varied slices
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ==============================
    # Model
    # ==============================
    if args.model == "unet":
        model = UNetBaseline(num_classes=3)
    else:
        model = TAM_UNet(num_classes=3)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    print("Starting inference...\n")

    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            if idx >= args.num_images:
                break

            images = images.to(device)
            masks = masks.to(device)

            # Debug info
            print("Unique mask values:", torch.unique(masks))

            if args.model == "tam":
                outputs, _ = model(images)
            else:
                outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            print("Unique prediction values:", torch.unique(preds))
            print("-" * 40)

            overlay_visualization(images[0], masks[0], preds[0])

    print("\nInference complete.")


if __name__ == "__main__":
    main()