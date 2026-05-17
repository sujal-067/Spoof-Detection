"""
app/main.py
─────────────────────────────────────────────────────────
Smart Depth Vision — Spoof Detection API  v2.0

Endpoints:
  POST /api/register         — new user account
  POST /api/login            — password auth → JWT
  POST /api/enroll-face      — add face images (multipart upload)
  POST /api/enroll-face-b64  — add face images (base64 webcam captures)
  POST /api/verify           — spoof check + face match
  POST /api/analyze-frame    — live webcam spoof analysis (no auth)
  GET  /api/logs             — fetch logs for logged-in user
  GET  /api/me               — current user info
  GET  /                     — serve the frontend SPA

Run:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Fixes applied vs original:
  - HTTPBearer used consistently (auth.py previously used OAuth2PasswordBearer
    pointing at a JSON endpoint, which broke Swagger Authorize)
  - SpoofLog now writes ALL fields (depth_std, clip_confidence,
    yolo_person_conf) that the DB schema defines — was only writing 5/9
  - get_models() failure at startup no longer kills the entire server
  - Frontend 404 returns clean JSON instead of crashing
"""

import base64
import json
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import (
    Depends, FastAPI, File, HTTPException,
    UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from app.auth import create_access_token, get_current_user, get_password_hash, verify_password
from app.database import SpoofLog, User, get_db, init_db
from app.face_utils import FaceUtils
from app.models_loader import get_models
from app.spoof_detector import SpoofDetector

# ── App setup ──────────────────────────────────────────────────────────────────

models = None 

app = FastAPI(
    title="Smart Depth Vision — Spoof Detection",
    description="Face anti-spoofing: YOLOv8 + MiDaS + CLIP + DeepFace",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend lives at  <repo_root>/frontend/index.html
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global models

    init_db()
    print("✓ Database initialised")

    try:
        models = get_models()
        print("✅ Models loaded successfully")
    except Exception as exc:
        print(f"❌ MODEL LOAD FAILED: {exc}")
        models = None


# ── Frontend ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": f"frontend/index.html not found at {index}"},
        )
    return FileResponse(index)


# ─── Global Error Handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("🔥 GLOBAL ERROR:", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    ) 


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class FrameRequest(BaseModel):
    frame: str   # data:image/jpeg;base64,...


class VerifyRequest(BaseModel):
    frame: str
    username: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_b64_image(b64_str: str) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    img_np    = np.frombuffer(img_bytes, np.uint8)
    frame     = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image from base64 string")
    return frame


def _resize_for_inference(frame: np.ndarray, max_w: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, int(h * scale)))
    return frame


# ── Auth endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/register", summary="Register a new user")
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    try:
        if len(req.username) < 3:
            raise HTTPException(status_code=400, detail="Username must be at least 3 characters")

        if len(req.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

        if db.query(User).filter(User.username == req.username).first():
            raise HTTPException(status_code=400, detail="Username already taken")

        if db.query(User).filter(User.email == req.email).first():
            raise HTTPException(status_code=400, detail="Email already registered")

        user = User(
            username=req.username,
            email=req.email,
            hashed_password=get_password_hash(req.password),
            face_embeddings=json.dumps([]),
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        token = create_access_token({"sub": user.username})

        return {
            "message": "Registered successfully. Now enroll your face.",
            "access_token": token,
            "token_type": "bearer",
            "user_id": user.id,
            "username": user.username,
            "face_enrolled": False,
        }

    except HTTPException:
        raise

    except Exception as e:
        print("❌ REGISTER ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/login", summary="Login with username + password → JWT")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Update last_login timestamp
    user.last_login = __import__("datetime").datetime.utcnow()
    db.commit()

    embeddings = json.loads(user.face_embeddings or "[]")
    token      = create_access_token({"sub": user.username})
    return {
        "access_token" : token,
        "token_type"   : "bearer",
        "user_id"      : user.id,
        "username"     : user.username,
        "face_enrolled": len(embeddings) > 0,
    }


@app.get("/api/me", summary="Current user info")
async def me(current_user: User = Depends(get_current_user)):
    embeddings = json.loads(current_user.face_embeddings or "[]")
    return {
        "user_id"      : current_user.id,
        "username"     : current_user.username,
        "email"        : current_user.email,
        "face_enrolled": len(embeddings) > 0,
        "face_count"   : len(embeddings),
        "registered_at": current_user.registered_at.isoformat(),
    }


# ── Face enrollment ────────────────────────────────────────────────────────────

@app.post("/api/enroll-face", summary="Upload 2-6 face images (multipart)")
async def enroll_face(
    images: List[UploadFile] = File(...),
    current_user: User       = Depends(get_current_user),
    db: Session              = Depends(get_db),
):
    if len(images) < 2:
        raise HTTPException(status_code=400, detail="Please upload at least 2 face images.")
    if len(images) > 6:
        raise HTTPException(status_code=400, detail="Maximum 6 images allowed.")

    face_utils     = FaceUtils()
    new_embeddings = []
    errors         = []

    for i, img_file in enumerate(images):
        raw    = await img_file.read()
        np_arr = np.frombuffer(raw, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            errors.append(f"Image {i+1}: Could not decode")
            continue
        try:
            emb = face_utils.extract_embedding(frame)
            new_embeddings.append(emb.tolist())
        except Exception as exc:
            errors.append(f"Image {i+1}: {exc}")

    if len(new_embeddings) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Could not detect a clear face in enough images. Errors: {errors}. "
                   "Please use well-lit, front-facing photos.",
        )

    existing       = json.loads(current_user.face_embeddings or "[]")
    all_embeddings = existing + new_embeddings
    db.query(User).filter(User.id == current_user.id).first().face_embeddings = json.dumps(all_embeddings)
    db.commit()

    return {
        "message"         : f"Face enrolled! {len(new_embeddings)} embedding(s) stored.",
        "total_embeddings": len(all_embeddings),
        "errors"          : errors,
    }


@app.post("/api/enroll-face-b64", summary="Enroll face via base64 webcam images")
async def enroll_face_b64(
    payload: dict,
    current_user: User = Depends(get_current_user),
    db: Session        = Depends(get_db),
):
    images_b64: List[str] = payload.get("images", [])
    if len(images_b64) < 2:
        raise HTTPException(status_code=400, detail="Please provide at least 2 images.")

    face_utils     = FaceUtils()
    new_embeddings = []
    errors         = []

    for i, b64 in enumerate(images_b64):
        try:
            frame = decode_b64_image(b64)
            emb   = face_utils.extract_embedding(frame)
            new_embeddings.append(emb.tolist())
        except Exception as exc:
            errors.append(f"Image {i+1}: {exc}")

    if len(new_embeddings) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Face detection failed on most images. {errors}",
        )

    existing       = json.loads(current_user.face_embeddings or "[]")
    all_embeddings = existing + new_embeddings
    db.query(User).filter(User.id == current_user.id).first().face_embeddings = json.dumps(all_embeddings)
    db.commit()

    return {
        "message"         : f"Face enrolled! {len(new_embeddings)} new embedding(s) added.",
        "total_embeddings": len(all_embeddings),
        "errors"          : errors,
    }


# ── Core analysis endpoints ───────────────────────────────────────────────────

@app.post("/api/analyze-frame")
async def analyze_frame(req: FrameRequest):
    global models

    if models is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        frame = decode_b64_image(req.frame)
        frame = _resize_for_inference(frame)

        detector = SpoofDetector(models)  # ✅ use preloaded models
        result = detector.analyze(frame)

        return JSONResponse(content=result)

    except Exception as exc:
        import traceback
        traceback.print_exc()  # 🔥 show real error
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/verify", summary="Full verification: spoof check + face identity match")
async def verify(req: VerifyRequest, db: Session = Depends(get_db)):
    """
    1. Spoof detection (YOLOv8 + MiDaS + CLIP)
    2. Face match against stored embeddings (only if not flagged as spoof)
    3. Write audit log entry with ALL score fields
    """
    try:
        frame    = decode_b64_image(req.frame)
        frame    = _resize_for_inference(frame)
        detector = SpoofDetector(models)

        # ── Step 1: spoof analysis ────────────────────────────────────────────
        analysis     = detector.analyze(frame)
        spoof_result = analysis.get("spoof_result", {})
        is_real      = spoof_result.get("is_real", False)

        # ── Step 2: face matching ─────────────────────────────────────────────
        face_matched  = False
        matched_user  = None
        face_distance = None

        if is_real:
            face_utils = FaceUtils()
            try:
                query_emb = face_utils.extract_embedding(frame)

                users = (
                    db.query(User).filter(User.username == req.username).all()
                    if req.username
                    else db.query(User).filter(User.is_active == True).all()
                )

                best_match    = None
                best_distance = float("inf")

                for user in users:
                    stored = json.loads(user.face_embeddings or "[]")
                    if not stored:
                        continue
                    dist = face_utils.match_embedding(
                        query_emb, [np.array(e) for e in stored]
                    )
                    if dist < best_distance:
                        best_distance = dist
                        best_match    = user

                THRESHOLD = 0.40
                if best_match and best_distance < THRESHOLD:
                    face_matched  = True
                    matched_user  = best_match.username
                    face_distance = round(float(best_distance), 4)

            except Exception as face_exc:
                print(f"[WARN] Face matching error: {face_exc}")

        # ── Step 3: write audit log (ALL fields) ──────────────────────────────
        log = SpoofLog(
            user_id          = None,
            depth_verdict    = spoof_result.get("depth_verdict"),
            depth_std        = spoof_result.get("depth_std"),          # ← was missing
            clip_verdict     = spoof_result.get("clip_verdict"),
            clip_confidence  = spoof_result.get("clip_score"),         # ← was missing
            yolo_person_conf = spoof_result.get("yolo_person_conf"),   # ← was missing
            combined_score   = spoof_result.get("combined_score"),
            is_spoof         = not is_real,
            face_matched     = face_matched,
        )
        if matched_user:
            user_obj = db.query(User).filter(User.username == matched_user).first()
            if user_obj:
                log.user_id = user_obj.id
        db.add(log)
        db.commit()

        return {
            "is_real"       : is_real,
            "face_matched"  : face_matched,
            "matched_user"  : matched_user,
            "face_distance" : face_distance,
            "access_granted": is_real and face_matched,
            "spoof_details" : spoof_result,
            "detections"    : analysis.get("yolo_detections", []),
            "depth_img"     : analysis.get("depth_img"),
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Verification error: {exc}")


@app.get("/api/logs", summary="Spoof attempt logs for current user")
async def get_logs(
    limit: int         = 20,
    current_user: User = Depends(get_current_user),
    db: Session        = Depends(get_db),
):
    logs = (
        db.query(SpoofLog)
        .filter(SpoofLog.user_id == current_user.id)
        .order_by(SpoofLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id"              : l.id,
            "timestamp"       : l.timestamp.isoformat(),
            "depth_verdict"   : l.depth_verdict,
            "depth_std"       : l.depth_std,
            "clip_verdict"    : l.clip_verdict,
            "clip_confidence" : l.clip_confidence,
            "yolo_person_conf": l.yolo_person_conf,
            "combined_score"  : l.combined_score,
            "is_spoof"        : l.is_spoof,
            "face_matched"    : l.face_matched,
        }
        for l in logs
    ]