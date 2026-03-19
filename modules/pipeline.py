import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from modules.depth import get_depth_map
from modules.detector import detect_objects
from modules.classifier import DepthAwareClassifier

IMG_SIZE = 128

rgb_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

depth_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

_classifier = None


def load_classifier(weights_path):
    global _classifier
    _classifier = DepthAwareClassifier(num_classes=2)
    checkpoint  = torch.load(weights_path, map_location="cpu")
    _classifier.load_state_dict(checkpoint["model_state"])
    _classifier.eval()
    print(f"[pipeline] Classifier loaded from {weights_path}")


def classify_roi(frame_rgb, depth_map, bbox, obj_label=""):
    """
    Classify a detected region as 2D or 3D using depth variance.
    Always returns exactly 3 values: (label, confidence, reason)

    High depth variance = 3D (real object)
    Low depth variance  = 2D (flat representation)
    """
    x1, y1, x2, y2 = bbox

    # Guard against tiny or invalid boxes
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return "unknown", 0.0, "box too small"

    # Crop depth map to bounding box
    depth_crop = depth_map[y1:y2, x1:x2].astype(np.float32)

    # Compute depth variance
    depth_mean = float(np.mean(depth_crop))
    depth_std  = float(np.std(depth_crop))

    # Coefficient of variation
    cv = depth_std / (depth_mean + 1e-6)

    # Edge density
    roi_rgb      = frame_rgb[y1:y2, x1:x2]
    gray         = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    edges        = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.mean())

    # Threshold
    DEPTH_THRESHOLD = 0.08

    if cv > DEPTH_THRESHOLD and edge_density < 90:
        label  = "3D"
        conf   = min(0.99, 0.5 + cv * 3)
        reason = f"depth std={depth_std:.1f}"
    else:
        label  = "2D"
        conf   = max(0.5, min(0.99,
                    0.5 + (DEPTH_THRESHOLD - cv) * 8 + 0.1))
        reason = f"flat depth std={depth_std:.1f}"

    return label, conf, reason


def process_frame(frame):
    """
    Full pipeline on one BGR frame.
    Returns annotated side-by-side frame (RGB | Depth).
    """
    frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth_map  = get_depth_map(frame)
    detections = detect_objects(frame)

    depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        obj_label       = det["label"]

        dim_label, conf, reason = classify_roi(
            frame_rgb, depth_map, det["bbox"], obj_label
        )

        color = (0, 200, 80) if dim_label == "3D" else (0, 80, 220)

        # Annotate RGB
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{obj_label} [{dim_label}] {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame,
                      (x1, y1 - th - 8), (x1 + tw + 4, y1),
                      color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)

        # Annotate depth
        cv2.rectangle(depth_color, (x1, y1), (x2, y2), color, 2)
        depth_crop = depth_map[y1:y2, x1:x2].astype(np.float32)
        avg_d = float(np.mean(depth_crop))
        std_d = float(np.std(depth_crop))

        dlabel = f"{obj_label} [{dim_label}]"
        (tw2, th2), _ = cv2.getTextSize(
            dlabel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(depth_color,
                      (x1, y1 - th2 - 6), (x1 + tw2 + 4, y1),
                      color, -1)
        cv2.putText(depth_color, dlabel, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.putText(depth_color,
                    f"avg:{avg_d:.0f} std:{std_d:.0f}",
                    (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 255, 200), 1)

    # Titles
    cv2.putText(frame, "RGB + Detection",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Green=3D  Blue=2D",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)
    cv2.putText(depth_color, "MiDaS Depth Map",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(depth_color, "Brighter = closer",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    # Stack side by side
    h1, w1 = frame.shape[:2]
    h2, w2 = depth_color.shape[:2]
    if h1 != h2:
        depth_color = cv2.resize(depth_color, (w2, h1))

    separator    = np.zeros((h1, 4, 3), dtype=np.uint8)
    separator[:] = (80, 80, 80)

    return np.hstack([frame, separator, depth_color])