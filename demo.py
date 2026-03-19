import cv2
import sys
import numpy as np
from modules.pipeline import load_classifier, process_frame
from modules.depth import get_depth_map
from modules.detector import detect_objects
from modules.pipeline import classify_roi

WEIGHTS_PATH = r"C:\Users\Dell\Desktop\PROJECT 1\Smart_Depth_Vision\weights\best_model.pth"

def annotate_both(frame):
    """
    Returns two annotated frames side by side:
      Left  — RGB with YOLO boxes + 2D/3D labels
      Right — Depth map with same YOLO boxes overlaid
    """
    import cv2
    import numpy as np

    frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth_map  = get_depth_map(frame)
    detections = detect_objects(frame)

    # ── Left: RGB annotated ──────────────────────────────
    rgb_out = frame.copy()

    # ── Right: Depth map as color image ─────────────────
    depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    for det in detections:
        x1, y1, x2, y2  = det["bbox"]
        obj_label        = det["label"]
        dim_label, conf  = classify_roi(frame_rgb, depth_map, det["bbox"])

        color = (0, 200, 80) if dim_label == "3D" else (0, 80, 220)

        # ── Draw on RGB frame ────────────────────────────
        cv2.rectangle(rgb_out, (x1, y1), (x2, y2), color, 2)
        text = f"{obj_label} [{dim_label}] {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(rgb_out,
                      (x1, y1 - th - 8), (x1 + tw + 4, y1),
                      color, -1)
        cv2.putText(rgb_out, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)

        # ── Draw same box on depth map ───────────────────
        cv2.rectangle(depth_color, (x1, y1), (x2, y2), color, 2)

        # Depth stats inside the box
        depth_crop = depth_map[y1:y2, x1:x2].astype(np.float32)
        avg_depth  = float(np.mean(depth_crop))
        std_depth  = float(np.std(depth_crop))

        # Label on depth map
        dlabel = f"{obj_label} [{dim_label}]"
        (tw2, th2), _ = cv2.getTextSize(
            dlabel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(depth_color,
                      (x1, y1 - th2 - 6), (x1 + tw2 + 4, y1),
                      color, -1)
        cv2.putText(depth_color, dlabel, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        # Depth variance info below box
        stats = f"avg:{avg_depth:.0f} std:{std_depth:.0f}"
        cv2.putText(depth_color, stats, (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 255, 200), 1)

    # ── Add titles ───────────────────────────────────────
    cv2.putText(rgb_out, "RGB + Detection",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(rgb_out, "Green=3D  Red=2D",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    cv2.putText(depth_color, "MiDaS Depth Map",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(depth_color, "Brighter = closer to camera",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    # ── Stack side by side ───────────────────────────────
    # Make sure both frames are same height
    h1, w1 = rgb_out.shape[:2]
    h2, w2 = depth_color.shape[:2]
    if h1 != h2:
        depth_color = cv2.resize(depth_color, (w2, h1))

    # Add a thin separator line
    separator = np.zeros((h1, 4, 3), dtype=np.uint8)
    separator[:] = (80, 80, 80)

    combined = np.hstack([rgb_out, separator, depth_color])
    return combined


def run_image(image_path):
    """Test pipeline on a single image."""
    load_classifier(WEIGHTS_PATH)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[demo] Cannot read image: {image_path}")
        sys.exit(1)

    print("[demo] Processing image...")
    combined = annotate_both(frame)

    # Resize if too wide for screen
    h, w = combined.shape[:2]
    max_w = 1400
    if w > max_w:
        scale    = max_w / w
        combined = cv2.resize(combined,
                              (int(w * scale), int(h * scale)))

    cv2.imshow("Smart Depth Vision — RGB | Depth", combined)
    print("[demo] Press any key to close  |  's' to save")

    key = cv2.waitKey(0) & 0xFF
    if key == ord('s'):
        out = "results/test_result.png"
        cv2.imwrite(out, combined)
        print(f"[demo] Saved → {out}")

    cv2.destroyAllWindows()


def run_webcam():
    """Live webcam demo."""
    load_classifier(WEIGHTS_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[demo] ERROR: Cannot open webcam.")
        sys.exit(1)

    print("[demo] Webcam running...")
    print("[demo] Press 'q' to quit  |  's' to save screenshot")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        combined = annotate_both(frame)

        # Resize if too wide
        h, w = combined.shape[:2]
        if w > 1400:
            scale    = 1400 / w
            combined = cv2.resize(combined,
                                  (int(w * scale), int(h * scale)))

        cv2.imshow("Smart Depth Vision — RGB | Depth", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            path = f"results/screenshot_{count:03d}.png"
            cv2.imwrite(path, combined)
            print(f"[demo] Screenshot saved → {path}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[demo] Done.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_image(sys.argv[1])
    else:
        run_webcam()