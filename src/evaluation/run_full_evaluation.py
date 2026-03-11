import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    jaccard_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

from src.dataset import LiverDataset
from src.models.unet_baseline import UNetBaseline
from src.models.tam_unet import TAM_UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Output folders
# ===============================

os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# ===============================
# Dataset
# ===============================

dataset = LiverDataset(
    images_dir="data/processed/images",
    masks_dir="data/processed/masks"
)

loader = DataLoader(
    dataset,
    batch_size=1,   # safer for memory
    shuffle=False
)

# ===============================
# Load models
# ===============================

baseline_model = UNetBaseline().to(device)
tam_model = TAM_UNet().to(device)

baseline_model.load_state_dict(torch.load("baseline_best_model.pth", map_location=device))
tam_model.load_state_dict(torch.load("tam_best_model.pth", map_location=device))

baseline_model.eval()
tam_model.eval()

# ===============================
# Prediction function
# ===============================

def get_predictions(model):

    all_preds = []
    all_probs = []
    all_gts = []

    with torch.no_grad():

        for images, masks in tqdm(loader):

            images = images.to(device)

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            preds = preds.cpu().numpy()
            probs = probs.cpu().numpy()
            gts = masks.numpy()

            # convert to tumor binary mask
            tumor_pred = (preds == 2).astype(int)
            tumor_gt = (gts == 2).astype(int)

            tumor_prob = probs[:,2,:,:]

            all_preds.append(tumor_pred.flatten())
            all_probs.append(tumor_prob.flatten())
            all_gts.append(tumor_gt.flatten())

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_gts = np.concatenate(all_gts)

    return all_preds, all_probs, all_gts


# ===============================
# Run evaluation
# ===============================

print("\nEvaluating Baseline U-Net\n")
pred_base, prob_base, gt = get_predictions(baseline_model)

print("\nEvaluating TAM U-Net\n")
pred_tam, prob_tam, _ = get_predictions(tam_model)

# ===============================
# Metrics
# ===============================

def compute_metrics(y_true, y_pred):

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    iou = jaccard_score(y_true, y_pred)

    dice = (2 * (y_true * y_pred).sum()) / (y_true.sum() + y_pred.sum() + 1e-8)

    return precision, recall, f1, accuracy, iou, dice


metrics_base = compute_metrics(gt, pred_base)
metrics_tam = compute_metrics(gt, pred_tam)

# ===============================
# Results table
# ===============================

results = pd.DataFrame({

    "Model": ["Baseline U-Net", "TAM-U-Net"],

    "Precision": [metrics_base[0], metrics_tam[0]],
    "Recall": [metrics_base[1], metrics_tam[1]],
    "F1": [metrics_base[2], metrics_tam[2]],
    "Accuracy": [metrics_base[3], metrics_tam[3]],
    "IoU": [metrics_base[4], metrics_tam[4]],
    "Dice": [metrics_base[5], metrics_tam[5]],

})

print("\nModel Performance Comparison\n")
print(results)

results.to_csv("outputs/metrics/model_comparison.csv", index=False)

# ===============================
# Dice comparison graph
# ===============================

plt.figure()

plt.bar(results["Model"], results["Dice"])

plt.ylabel("Dice Score")
plt.title("Dice Score Comparison")

plt.savefig("outputs/plots/dice_comparison.png")
plt.close()

# ===============================
# Confusion Matrix
# ===============================

def plot_confusion(y_true, y_pred, title, filename):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()

    sns.heatmap(cm, annot=True, fmt="d")

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"outputs/plots/{filename}")
    plt.close()


plot_confusion(gt, pred_base, "Baseline U-Net Confusion Matrix", "baseline_confusion.png")
plot_confusion(gt, pred_tam, "TAM U-Net Confusion Matrix", "tam_confusion.png")

# ===============================
# ROC Curve
# ===============================

plt.figure()

fpr1, tpr1, _ = roc_curve(gt, prob_base)
fpr2, tpr2, _ = roc_curve(gt, prob_tam)

auc1 = auc(fpr1, tpr1)
auc2 = auc(fpr2, tpr2)

plt.plot(fpr1, tpr1, label=f"Baseline U-Net (AUC={auc1:.3f})")
plt.plot(fpr2, tpr2, label=f"TAM-U-Net (AUC={auc2:.3f})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("outputs/plots/roc_curve.png")
plt.close()

# ===============================
# Precision Recall Curve
# ===============================

plt.figure()

p1, r1, _ = precision_recall_curve(gt, prob_base)
p2, r2, _ = precision_recall_curve(gt, prob_tam)

plt.plot(r1, p1, label="Baseline U-Net")
plt.plot(r2, p2, label="TAM-U-Net")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")
plt.legend()

plt.savefig("outputs/plots/pr_curve.png")
plt.close()

print("\nEvaluation complete.")
print("Metrics saved to outputs/metrics/")
print("Plots saved to outputs/plots/")