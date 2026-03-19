import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from modules.dataset import SmartDepthDataset
from modules.classifier import DepthAwareClassifier
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────
BASE_DIR    = r"C:\Users\Dell\Desktop\PROJECT 1\Smart_Depth_Vision\data"
WEIGHTS_DIR = r"C:\Users\Dell\Desktop\PROJECT 1\Smart_Depth_Vision\weights"
EPOCHS      = 20     # increase to 20 for final submission
BATCH_SIZE  = 8
LR          = 1e-4
VAL_SPLIT   = 0.2
MAX_3D      = 1449    # increase to 1449 for final submission
MAX_2D      = None    # increase to None for final submission
# ────────────────────────────────────────────────────────

os.makedirs(WEIGHTS_DIR, exist_ok=True)
device = torch.device("cpu")
print(f"[train] Using device: {device}")
print(f"[train] Config → Epochs: {EPOCHS} | Max 3D: {MAX_3D} | Max 2D: {MAX_2D}")

# Dataset
print("[train] Loading dataset...")
dataset = SmartDepthDataset(BASE_DIR, max_2d=MAX_2D, max_3d=MAX_3D)

if len(dataset) == 0:
    raise RuntimeError("Dataset is empty! Check your data folders.")

n_val   = max(1, int(len(dataset) * VAL_SPLIT))
n_train = len(dataset) - n_val
train_set, val_set = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

print(f"[train] Train samples: {n_train} | Val samples: {n_val}")

# Model
model     = DepthAwareClassifier(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

best_val_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_acc": []}

print("[train] Starting training...\n")

for epoch in range(1, EPOCHS + 1):

    # ── Train phase ──────────────────────────────────────
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for rgb, depth, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch:02d}/{EPOCHS} [train]",
                                   leave=False):
        rgb    = rgb.to(device)
        depth  = depth.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(rgb, depth)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * labels.size(0)
        preds          = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

    scheduler.step()
    train_acc  = 100.0 * train_correct / train_total
    train_loss /= train_total

    # ── Validation phase ─────────────────────────────────
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for rgb, depth, labels in val_loader:
            rgb    = rgb.to(device)
            depth  = depth.to(device)
            labels = labels.to(device)
            outputs = model(rgb, depth)
            preds   = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = 100.0 * val_correct / val_total

    # ── Logging ──────────────────────────────────────────
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.1f}% | "
          f"Val Acc: {val_acc:.1f}%", end="")

    # ── Save best model ───────────────────────────────────
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(WEIGHTS_DIR, "best_model.pth")
        torch.save({
            "epoch":      epoch,
            "model_state": model.state_dict(),
            "val_acc":    val_acc,
            "train_acc":  train_acc,
        }, save_path)
        print(f"  ✓ saved (best so far)", end="")

    print()  # newline

# ── Final summary ────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Training complete!")
print(f"  Best val accuracy : {best_val_acc:.1f}%")
print(f"  Weights saved to  : {WEIGHTS_DIR}\\best_model.pth")
print(f"{'='*55}")

# ── Save training history ────────────────────────────────
import json
history_path = os.path.join(WEIGHTS_DIR, "history.json")
with open(history_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"  History saved to  : {history_path}")