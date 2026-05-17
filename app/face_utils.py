"""
app/face_utils.py
─────────────────────────────────────────────────────────
Face embedding extraction and matching using DeepFace (Facenet model).

Fixes applied:
  - list[np.ndarray] type hint is Python 3.10+ only — replaced with
    List[np.ndarray] from typing (works on Python 3.8+)

Why DeepFace:
  • Works on Windows without compiling dlib
  • Facenet gives 128-dim embeddings with good accuracy
  • Simple API — no GPU required (runs on CPU)

Flow:
  Enrollment  : extract_embedding(image) → numpy array → store as list in DB
  Verification: match_embedding(query, stored_list) → cosine distance → compare threshold
"""

import cv2
import numpy as np
from typing import List   # ← use typing.List for Python 3.8/3.9 compatibility


class FaceUtils:
    """
    Wrapper around DeepFace for face embedding extraction and comparison.
    Model is loaded lazily on first use.
    """

    MODEL_NAME = "Facenet"    # 128-dim, fast, accurate
    DETECTOR   = "opencv"     # retinaface or mtcnn for higher accuracy;
    #                           opencv is fastest and most compatible

    def extract_embedding(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extract a 128-dim face embedding from a BGR image.
        Raises ValueError if no face is found.

        Args:
            frame_bgr: OpenCV BGR image

        Returns:
            np.ndarray of shape (128,) — L2 normalized embedding
        """
        try:
            from deepface import DeepFace
        except ImportError:
            raise ImportError(
                "DeepFace is not installed. Run: pip install deepface"
            )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        try:
            result = DeepFace.represent(
                img_path=frame_rgb,
                model_name=self.MODEL_NAME,
                detector_backend=self.DETECTOR,
                enforce_detection=True,
                align=True,
            )
        except Exception as e:
            raise ValueError(
                f"No clear face detected in the image. "
                f"Please use a well-lit, front-facing photo. (Detail: {e})"
            )

        # result is a list of dicts; take the highest-confidence face
        if isinstance(result, list):
            result = result[0]

        embedding = np.array(result["embedding"], dtype=np.float32)

        # L2 normalize so cosine distance = 1 - dot-product
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        return embedding

    def match_embedding(
        self,
        query: np.ndarray,
        stored: List[np.ndarray],      # ← was list[...] (3.10+ only), now List[...]
        aggregation: str = "min",
    ) -> float:
        """
        Compute the best cosine distance between query and stored embeddings.

        Args:
            query       : (128,) normalized embedding of the new face
            stored      : list of (128,) normalized embeddings from enrollment
            aggregation : "min" uses the best match, "mean" uses average

        Returns:
            float: cosine distance in [0, 2].
            Lower = more similar. Typical threshold: < 0.40 = same person.
        """
        if not stored:
            return float("inf")

        distances = []
        for emb in stored:
            emb  = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb = emb / norm
            cos_sim  = float(np.dot(query, emb))
            cos_dist = 1.0 - cos_sim
            distances.append(cos_dist)

        if aggregation == "mean":
            return float(np.mean(distances))
        return min(distances)   # default: "min"

    @staticmethod
    def distance_to_confidence(distance: float, threshold: float = 0.40) -> float:
        """
        Convert cosine distance to match confidence (0–100%).
          distance = 0          → 100%
          distance = threshold  → 50%
          distance ≥ threshold*2→ 0%
        """
        if distance <= 0:
            return 100.0
        if distance >= threshold * 2:
            return 0.0
        return round(100.0 * (1.0 - distance / (threshold * 2)), 1)