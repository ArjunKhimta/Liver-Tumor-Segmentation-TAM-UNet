import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from dataset import LiverDataset
from models.tam_unet import TAM_UNet


# ==============================
# CONFIGURATION
# ==============================
WEIGHTS_PATH = "tam_best_model.pth"
VAL_SPLIT_RATIO = 0.2
MAX_SLICES = 500
THRESHOLD = 0.5
NUM_CLASSES = 3

# ==============================
# DEVICE
# ==============================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
# LOAD DATASET
# ==============================
IMAGES_DIR = "data/processed/images"
MASKS_DIR = "data/processed/masks"

dataset = LiverDataset(IMAGES_DIR, MASKS_DIR)
n_total = len(dataset)

# Validation split (last 20%)
val_start = int((1 - VAL_SPLIT_RATIO) * n_total)
val_indices = list(range(val_start, n_total))
val_dataset = Subset(dataset, val_indices)

if MAX_SLICES < len(val_dataset):
    val_dataset = Subset(val_dataset, range(MAX_SLICES))

loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print(f"Validation slices used: {len(loader)}")

# ==============================
# LOAD MODEL
# ==============================
model = TAM_UNet(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device)
model.eval()

all_probs = []
all_labels = []

print("Running validation evaluation...")

# ==============================
# COLLECT PREDICTIONS
# ==============================
with torch.no_grad():
    for images, masks in tqdm(loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs, _ = model(images)
        probs = torch.softmax(outputs, dim=1)

        # Convert to (B,H,W,C)
        probs = probs.permute(0, 2, 3, 1).cpu().numpy()
        masks = masks.cpu().numpy()

        # Flatten per pixel
        all_probs.append(probs.reshape(-1, NUM_CLASSES))
        all_labels.append(masks.flatten())

# Concatenate everything
all_probs = np.concatenate(all_probs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print("Total evaluated pixels:", len(all_labels))

# ==============================
# MULTI-CLASS CONFUSION MATRIX
# ==============================
pred_classes = np.argmax(all_probs, axis=1)
cm_multi = confusion_matrix(all_labels, pred_classes)

print("\nMulti-Class Confusion Matrix (3x3):")
print(cm_multi)

# ==============================
# MULTI-CLASS ROC
# ==============================
labels_bin = label_binarize(all_labels, classes=[0, 1, 2])

plt.figure()
for i, class_name in enumerate(["Background", "Liver", "Tumor"]):
    fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    print(f"{class_name} AUC: {roc_auc:.4f}")
    plt.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.3f})")

plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve")
plt.legend()
plt.savefig("multiclass_roc.png")
plt.show()

# ==============================
# MULTI-CLASS PRECISION-RECALL
# ==============================
plt.figure()
for i, class_name in enumerate(["Background", "Liver", "Tumor"]):
    precision, recall, _ = precision_recall_curve(labels_bin[:, i], all_probs[:, i])
    plt.plot(recall, precision, label=class_name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Multi-Class Precision-Recall Curve")
plt.legend()
plt.savefig("multiclass_pr.png")
plt.show()

# ==============================
# TUMOR BINARY METRICS
# ==============================
tumor_probs = all_probs[:, 2]
tumor_labels = (all_labels == 2).astype(int)

pred_binary = (tumor_probs > THRESHOLD).astype(int)

f1 = f1_score(tumor_labels, pred_binary)
cm_binary = confusion_matrix(tumor_labels, pred_binary)

tn, fp, fn, tp = cm_binary.ravel()

sensitivity = tp / (tp + fn + 1e-8)
specificity = tn / (tn + fp + 1e-8)

print("\nTumor Binary Confusion Matrix:")
print(cm_binary)

print(f"\nTumor F1 Score: {f1:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")