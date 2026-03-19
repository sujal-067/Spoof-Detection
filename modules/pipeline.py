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
def classify_roi(frame_rgb, depth_map, bbox):
    """
    Instead of classifying just the crop,
    we use depth statistics of the ROI to determine 2D vs 3D.
    Real 3D objects have HIGH depth variance.
    Flat 2D objects have LOW depth variance.
    """
    x1, y1, x2, y2 = bbox

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return "unknown", 0.0

    # Crop depth map to bounding box
    depth_crop = depth_map[y1:y2, x1:x2].astype(np.float32)

    # Compute depth variance in this region
    depth_std  = float(np.std(depth_crop))
    depth_mean = float(np.mean(depth_crop))

    # Normalize variance relative to mean (coefficient of variation)
    cv = depth_std / (depth_mean + 1e-6)

    # Threshold: high variance = 3D, low variance = 2D
    # Tuned for MiDaS small output
    # Threshold tuned for MiDaS small on indoor scenes
    THRESHOLD = 0.08   # was 0.15 — lower = more sensitive to 3D

    if cv > THRESHOLD:
        label = "3D"
        # Scale confidence: higher variance = more confident
        conf  = min(0.99, 0.5 + cv * 3)
    else:
        label = "2D"
        conf  = min(0.99, 0.5 + (THRESHOLD - cv) * 8)

    return label, conf

def process_frame(frame):
    """
    Full pipeline on one BGR frame:
      1. Detect objects with YOLOv8
      2. Generate depth map with MiDaS
      3. Classify each detection as 2D or 3D
      4. Annotate and return the frame
    """
    frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth_map  = get_depth_map(frame)          # uint8 HxW
    detections = detect_objects(frame)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        obj_label       = det["label"]
        dim_label, conf = classify_roi(frame_rgb, depth_map, det["bbox"])

        # Color: green = 3D, red = 2D
        color = (0, 200, 80) if dim_label == "3D" else (0, 80, 220)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        text  = f"{obj_label} [{dim_label}] {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Overlay depth map (top-right corner, small)
    h, w   = frame.shape[:2]
    dsize  = (w // 4, h // 4)
    depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    depth_small = cv2.resize(depth_color, dsize)
    frame[10:10+dsize[1], w-dsize[0]-10:w-10] = depth_small
    cv2.putText(frame, "depth", (w - dsize[0] - 10, dsize[1] + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return frame