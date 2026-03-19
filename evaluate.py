import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             accuracy_score,
                             f1_score)
from modules.dataset import SmartDepthDataset
from modules.classifier import DepthAwareClassifier
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────
BASE_DIR     = r"C:\Users\Dell\Desktop\PROJECT 1\Smart_Depth_Vision\data"
WEIGHTS_PATH = r"C:\Users\Dell\Desktop\PROJECT 1\Smart_Depth_Vision\weights\best_model.pth"
RESULTS_DIR  = r"C:\Users\Dell\Desktop\PROJECT 1\Smart_Depth_Vision\results"
MAX_3D       = 1449
MAX_2D       = None
BATCH_SIZE   = 8
CLASS_NAMES  = ["2D", "3D"]
# ────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("cpu")

# ── Load dataset (same split as training) ───────────────
print("[eval] Loading dataset...")
dataset = SmartDepthDataset(BASE_DIR, max_2d=MAX_2D, max_3d=MAX_3D)

n_val   = max(1, int(len(dataset) * 0.2))
n_train = len(dataset) - n_val
_, val_set = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
print(f"[eval] Evaluating on {n_val} validation samples...")

# ── Load model ───────────────────────────────────────────
print("[eval] Loading model weights...")
model = DepthAwareClassifier(num_classes=2).to(device)
checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"[eval] Loaded checkpoint from epoch {checkpoint['epoch']} "
      f"(val acc: {checkpoint['val_acc']:.1f}%)")

# ── Run inference ────────────────────────────────────────
all_preds  = []
all_labels = []

with torch.no_grad():
    for rgb, depth, labels in tqdm(val_loader, desc="[eval] Inference"):
        rgb    = rgb.to(device)
        depth  = depth.to(device)
        outputs = model(rgb, depth)
        preds   = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Metrics ──────────────────────────────────────────────
acc = accuracy_score(all_labels, all_preds) * 100
f1  = f1_score(all_labels, all_preds, average="weighted") * 100

print(f"\n{'='*55}")
print(f"  Accuracy  : {acc:.2f}%")
print(f"  F1 Score  : {f1:.2f}%")
print(f"{'='*55}")
print("\n[eval] Classification Report:")
print(classification_report(all_labels, all_preds,
                             target_names=CLASS_NAMES))

# ── Save report to txt ───────────────────────────────────
report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"Accuracy : {acc:.2f}%\n")
    f.write(f"F1 Score : {f1:.2f}%\n\n")
    f.write(classification_report(all_labels, all_preds,
                                   target_names=CLASS_NAMES))
print(f"\n[eval] Report saved → {report_path}")

# ── Confusion matrix ─────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=CLASS_NAMES)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Smart Depth Vision", fontsize=13)
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"[eval] Confusion matrix saved → {cm_path}")

# ── Training history plot ────────────────────────────────
history_path = r"C:\Users\Dell\Desktop\PROJECT 1\Smart_Depth_Vision\weights\history.json"
if os.path.exists(history_path):
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax1.plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    ax1.plot(epochs, history["val_acc"],   "g-o", label="Val Acc")
    ax1.set_title("Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax1.grid(True)

    # Loss plot
    ax2.plot(epochs, history["train_loss"], "r-o", label="Train Loss")
    ax2.set_title("Loss over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle("Smart Depth Vision — Training History", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "training_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[eval] Training history plot saved → {plot_path}")

print(f"\n[eval] All results saved to: {RESULTS_DIR}")