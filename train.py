import torch
import random
import numpy as np
import argparse
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn

from src.dataset import LiverDataset
from src.losses import CombinedLoss
from src.models.unet_baseline import UNetBaseline
from src.models.tam_unet import TAM_UNet


# ==============================
# Reproducibility
# ==============================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ==============================
# Dice Score (Multi-class)
# ==============================
def dice_score(preds, targets, num_classes=3):
    dice_scores = []

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        dice = (2. * intersection) / (union + 1e-5)
        dice_scores.append(dice.item())

    return dice_scores


# ==============================
# Main Training Function
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "tam"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)

    args = parser.parse_args()

    # ==============================
    # Device
    # ==============================
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 🔥 Safe MPS speed improvement
    if device.type == "mps":
        torch.set_float32_matmul_precision("high")

    print(f"\nUsing device: {device}")
    print(f"Training model: {args.model}\n")

    # ==============================
    # Dataset
    # ==============================
    IMAGES_DIR = "data/processed/images"
    MASKS_DIR = "data/processed/masks"

    full_dataset = LiverDataset(IMAGES_DIR, MASKS_DIR)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    # 🔥 Faster DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")

    # ==============================
    # Model
    # ==============================
    if args.model == "unet":
        model = UNetBaseline(num_classes=3).to(device)
    else:
        model = TAM_UNet(num_classes=3).to(device)
    # Class weights (tumor emphasized)
    weights = torch.tensor([0.2, 0.3, 1.5]).to(device)
    criterion = CombinedLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    aux_criterion = nn.BCEWithLogitsLoss()

    # ==============================
    # Early Stopping Setup
    # ==============================
    best_tumor_dice = 0
    patience_counter = 0

    print("Starting training...\n")

    total_start_time = time.time()

    # ==============================
    # Training Loop
    # ==============================
    for epoch in range(args.epochs):

        epoch_start_time = time.time()
        print(f"\n========== Epoch {epoch+1}/{args.epochs} ==========")

        # -------- TRAIN --------
        model.train()
        train_loss = 0

        train_bar = tqdm(train_loader, desc="Training", leave=False)

        for images, masks in train_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            if args.model == "tam":
                main_out, aux_out = model(images)

                main_loss = criterion(main_out, masks)

                tumor_mask = (masks == 2).float().unsqueeze(1)
                aux_loss = aux_criterion(aux_out, tumor_mask)

                loss = main_loss + 0.3 * aux_loss
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        total_dice = [0, 0, 0]

        val_bar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for images, masks in val_bar:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                if args.model == "tam":
                    outputs, _ = model(images)
                else:
                    outputs = model(images)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                dice_scores = dice_score(preds, masks)

                for i in range(3):
                    total_dice[i] += dice_scores[i]

        avg_val_loss = val_loss / len(val_loader)
        avg_dice = [d / len(val_loader) for d in total_dice]

        # -------- Time Tracking --------
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - total_start_time

        avg_epoch_time = total_time / (epoch + 1)
        estimated_total_time = avg_epoch_time * args.epochs
        estimated_remaining = estimated_total_time - total_time

        # -------- Logging --------
        print(f"\nEpoch [{epoch+1}/{args.epochs}] Completed")
        print(f"Epoch Time: {epoch_time/60:.2f} min")
        print(f"Estimated Remaining Time: {estimated_remaining/60:.2f} min")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Dice - Background: {avg_dice[0]:.4f}")
        print(f"Dice - Liver: {avg_dice[1]:.4f}")
        print(f"Dice - Tumor: {avg_dice[2]:.4f}")

        # -------- Early Stopping --------
        if avg_dice[2] > best_tumor_dice:
            best_tumor_dice = avg_dice[2]
            patience_counter = 0

            save_name = (
                "tam_best_model.pth"
                if args.model == "tam"
                else "baseline_best_model.pth"
            )

            torch.save(model.state_dict(), save_name)
            print("🔥 Best model saved!\n")

        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{args.patience})\n")

            if patience_counter >= args.patience:
                print("🛑 Early stopping triggered.")
                break

    print("\nTraining Complete.")
    print(f"Best Tumor Dice: {best_tumor_dice:.4f}")


if __name__ == "__main__":
    main()