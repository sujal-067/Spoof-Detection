import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from modules.depth import get_or_cache_depth

IMG_SIZE = 128  # small = faster on CPU

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


class SmartDepthDataset(Dataset):
    """
    Loads:
      - NYU images (data/extracted/images/) with pre-existing depths
        (data/extracted/depths/) → label 1 (3D)
      - COCO/flat images (data/2d_images/) with MiDaS-generated depths
        cached in (data/depth_cache/) → label 0 (2D)
    """

    def __init__(self, base_dir, max_2d=None, max_3d=None):
        self.samples = []  # list of (rgb_path, depth_path_or_none, label)
        self.cache_dir = os.path.join(base_dir, "depth_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # --- 3D samples: NYU ---
        nyu_rgb_dir   = os.path.join(base_dir, "extracted", "images")
        nyu_depth_dir = os.path.join(base_dir, "extracted", "depths")

        nyu_files = sorted([
            f for f in os.listdir(nyu_rgb_dir)
            if f.endswith(".png") or f.endswith(".jpg")
        ])
        if max_3d:
            nyu_files = nyu_files[:max_3d]

        for fname in nyu_files:
            rgb_path   = os.path.join(nyu_rgb_dir, fname)
            depth_path = os.path.join(nyu_depth_dir, fname)
            if os.path.exists(depth_path):
                self.samples.append((rgb_path, depth_path, 1))

        print(f"[dataset] 3D samples (NYU): {sum(1 for _,_,l in self.samples if l==1)}")

        # --- 2D samples: COCO/flat images ---
        flat_dir = os.path.join(base_dir, "2d_images", "val2017")
        flat_files = []
        if os.path.exists(flat_dir):
            flat_files = sorted([
                f for f in os.listdir(flat_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
        if max_2d:
            flat_files = flat_files[:max_2d]

        for fname in flat_files:
            rgb_path = os.path.join(flat_dir, fname)
            self.samples.append((rgb_path, None, 0))  # depth generated on-the-fly

        print(f"[dataset] 2D samples (flat): {sum(1 for _,_,l in self.samples if l==0)}")
        print(f"[dataset] Total: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, label = self.samples[idx]

        # Load RGB
        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb_tensor = rgb_transform(rgb_img)

        # Load or generate depth
        if depth_path is not None:
            # NYU pre-existing depth (16-bit PNG → convert to 8-bit)
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            depth_8 = cv2.normalize(depth_raw, None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
        else:
            # 2D image: generate with MiDaS, cached to disk
            depth_8 = get_or_cache_depth(rgb_path, self.cache_dir)

        depth_pil = Image.fromarray(depth_8, mode="L")
        depth_tensor = depth_transform(depth_pil)

        return rgb_tensor, depth_tensor, torch.tensor(label, dtype=torch.long)