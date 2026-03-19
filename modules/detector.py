from ultralytics import YOLO

_model = None

def _load_model():
    global _model
    if _model is None:
        print("[detector] Loading YOLOv8n...")
        _model = YOLO("yolov8n.pt")
        print("[detector] YOLOv8n loaded.")

def detect_objects(frame):
    """
    Returns list of dicts:
      { bbox: (x1,y1,x2,y2), label: str, conf: float }
    """
    _load_model()
    results = _model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results.names[int(box.cls[0])]
        conf  = float(box.conf[0])
        if conf > 0.25:
            detections.append({
                "bbox":  (x1, y1, x2, y2),
                "label": label,
                "conf":  conf
            })
    return detections