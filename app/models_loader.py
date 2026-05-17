"""
app/models_loader.py
─────────────────────────────────────────────────────────
Singleton loader for all heavy ML models.
Called once at startup — subsequent calls return the cached dict.

Models loaded:
  • YOLOv8s        — object detection (ultralytics)
  • MiDaS DPT_Hybrid — monocular depth estimation (torch.hub)
  • CLIP ViT-B/32  — image-text similarity for spoof classification

Fixes applied:
  - dict | None  is Python 3.10+ syntax — replaced with Optional[Dict]  ← FIXED
  - import clip appeared twice — removed duplicate                        ← FIXED
  - config.yaml missing crashes the whole server — added safe defaults    ← FIXED
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml

_models: Optional[Dict] = None   # module-level cache


def get_models() -> Dict:
    """Return the loaded model bundle, loading once on first call."""
    global _models
    if _models is None:
        _models = _load_all()
    return _models


# ── Default config values (used when config.yaml is missing / incomplete) ──────
_DEFAULTS = {
    "models": {
        "yolo": {
            "variant": "yolov8s.pt",   # auto-downloads ~22 MB on first run
            "conf":    0.30,
        }
    }
}


def _load_cfg() -> Dict:
    """Load config.yaml from project root, falling back to defaults gracefully."""
    root     = Path(__file__).resolve().parent.parent
    cfg_path = root / "config.yaml"
    if not cfg_path.exists():
        print(f"[models_loader] config.yaml not found at {cfg_path} — using defaults")
        return _DEFAULTS

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}

    # Merge with defaults so missing keys don't cause KeyError later
    cfg.setdefault("models", {})
    cfg["models"].setdefault("yolo", {})
    cfg["models"]["yolo"].setdefault("variant", _DEFAULTS["models"]["yolo"]["variant"])
    cfg["models"]["yolo"].setdefault("conf",    _DEFAULTS["models"]["yolo"]["conf"])
    return cfg


def _load_all() -> Dict:
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    cfg    = _load_cfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[models_loader] device = {device}")

    # ── YOLOv8 ────────────────────────────────────────────────────────────────
    from ultralytics import YOLO
    yolo_variant = cfg["models"]["yolo"]["variant"]
    yolo = YOLO(yolo_variant)
    print(f"  ✓ YOLOv8 ({yolo_variant}) loaded")

    # ── MiDaS DPT_Hybrid ──────────────────────────────────────────────────────
    midas = torch.hub.load(
        "intel-isl/MiDaS", "DPT_Hybrid",
        pretrained=True, trust_repo=True,
    )
    midas = midas.to(device).eval()
    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS", "transforms", trust_repo=True,
    )
    midas_transform = midas_transforms.dpt_transform
    print("  ✓ MiDaS DPT_Hybrid loaded")

    # ── CLIP ──────────────────────────────────────────────────────────────────
    try:
        import clip as openai_clip   # imported once — no duplicate below
    except ImportError:
        raise ImportError(
            "openai-clip not found. Install it with:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device=device)
    clip_model.eval()
    print("  ✓ CLIP ViT-B/32 loaded")

    # ── Pre-encode CLIP text prompts (done once, reused every frame) ──────────
    REAL_PROMPTS = [
        "a real human face looking at the camera",
        "a live person's face",
        "a genuine living human face",
    ]
    SPOOF_PROMPTS = [
        "a printed photograph of a face",
        "a face displayed on a phone or monitor screen",
        "a face mask or mannequin",
        "a flat 2D image of a face",
    ]

    with torch.no_grad():
        real_tokens  = openai_clip.tokenize(REAL_PROMPTS).to(device)
        spoof_tokens = openai_clip.tokenize(SPOOF_PROMPTS).to(device)
        real_text_feat  = clip_model.encode_text(real_tokens).float()
        spoof_text_feat = clip_model.encode_text(spoof_tokens).float()
        # Average the ensemble prompts for each class
        real_text_feat  = real_text_feat.mean(dim=0, keepdim=True)
        spoof_text_feat = spoof_text_feat.mean(dim=0, keepdim=True)
        # L2 normalize
        real_text_feat  = real_text_feat  / real_text_feat.norm(dim=-1, keepdim=True)
        spoof_text_feat = spoof_text_feat / spoof_text_feat.norm(dim=-1, keepdim=True)

    print("  ✓ CLIP text embeddings pre-computed")
    print(f"\n[models_loader] All models ready on [{device}]\n")

    return {
        "yolo":            yolo,
        "midas":           midas,
        "midas_transform": midas_transform,
        "clip_model":      clip_model,
        "clip_preprocess": clip_preprocess,
        "real_text_feat":  real_text_feat,
        "spoof_text_feat": spoof_text_feat,
        "device":          device,
        "cfg":             cfg,
    }