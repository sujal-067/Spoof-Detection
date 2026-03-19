import torch
import cv2
import numpy as np
from PIL import Image
import os

# Load MiDaS small (CPU-friendly)
_model = None
_transform = None

def _load_model():
    global _model, _transform
    if _model is None:
        print("[depth] Loading MiDaS small model...")
        _model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        _model.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        _transform = transforms.small_transform
        print("[depth] MiDaS loaded.")

def get_depth_map(image_input):
    """
    Accepts either:
      - a file path (str)
      - a numpy BGR frame directly (from cv2.imread or webcam)
    Returns a normalized uint8 depth map.
    """
    _load_model()

    # Handle both file path and numpy array
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Already a numpy BGR frame from cv2
        img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

    input_batch = _transform(img_rgb)
    with torch.no_grad():
        prediction = _model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.numpy()
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    return depth_norm.astype(np.uint8)


def get_or_cache_depth(image_path, cache_dir):
    """
    Returns depth map as uint8 array.
    If cached PNG exists, loads it. Otherwise runs MiDaS and saves to cache.
    """
    image_path = str(image_path)
    fname = os.path.splitext(os.path.basename(image_path))[0] + ".png"
    cache_path = os.path.join(cache_dir, fname)

    if os.path.exists(cache_path):
        depth = cv2.imread(cache_path, cv2.IMREAD_GRAYSCALE)
        return depth

    depth = get_depth_map(image_path)
    os.makedirs(cache_dir, exist_ok=True)
    cv2.imwrite(cache_path, depth)
    return depth