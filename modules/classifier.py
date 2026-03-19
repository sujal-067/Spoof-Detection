import torch
import torch.nn as nn
from torchvision import models

class DepthAwareClassifier(nn.Module):
    """
    Dual-stream MobileNetV2:
      Stream 1 → RGB features (512-d)
      Stream 2 → Depth features (512-d)
      Fusion   → concat → FC → 2 classes (2D / 3D)
    """

    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()

        # RGB stream
        rgb_base = models.mobilenet_v2(weights="DEFAULT")
        rgb_base.classifier = nn.Identity()
        self.rgb_stream = rgb_base

        # Depth stream (same arch, different weights)
        depth_base = models.mobilenet_v2(weights="DEFAULT")
        # Adapt first conv to accept 1-channel depth
        depth_base.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        depth_base.classifier = nn.Identity()
        self.depth_stream = depth_base

        # Fusion head
        # MobileNetV2 outputs 1280-d after AdaptiveAvgPool
        self.fusion = nn.Sequential(
            nn.Linear(1280 + 1280, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb, depth):
        # MobileNetV2 without classifier returns (B, 1280, 1, 1)
        # We need to flatten
        f_rgb   = self.rgb_stream.features(rgb)
        f_rgb   = f_rgb.mean([2, 3])          # global avg pool → (B, 1280)

        f_depth = self.depth_stream.features(depth)
        f_depth = f_depth.mean([2, 3])        # → (B, 1280)

        fused = torch.cat([f_rgb, f_depth], dim=1)  # (B, 2560)
        return self.fusion(fused)