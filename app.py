import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.depth import get_depth_map
from modules.detector import detect_objects
from modules.pipeline import load_classifier, classify_roi

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Smart Depth Vision",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%);
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050510 0%, #0a1628 100%);
        border-right: 1px solid #1a3a5c;
    }
    [data-testid="stSidebar"] * { color: #c0d8f0 !important; }
    [data-testid="stHeader"] {
        background: rgba(5, 5, 20, 0.95);
        border-bottom: 1px solid #1a3a5c;
    }
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00C9FF, #92FE9D, #00C9FF);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        margin-bottom: 0;
    }
    @keyframes shine { to { background-position: 200% center; } }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #7090b0;
        margin-top: 4px;
        letter-spacing: 0.5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #0d1f35, #0a1628);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        border: 1px solid #1a4a7a;
        box-shadow: 0 4px 20px rgba(0,100,200,0.15);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); border-color: #00C9FF; }
    .metric-card h2 { color: #00C9FF; font-size: 2.2rem; margin: 0; }
    .metric-card p  { color: #7090b0; margin: 4px 0 0 0; }
    .team-card {
        background: linear-gradient(135deg, #0d1f35, #0a1628);
        border-radius: 12px;
        padding: 18px 22px;
        margin: 10px 0;
        border-left: 4px solid #00C9FF;
        box-shadow: 0 2px 15px rgba(0,100,200,0.1);
        color: #c0d8f0;
    }
    .team-lead {
        border-left: 4px solid #92FE9D;
        background: linear-gradient(135deg, #0a2a1a, #0a1628);
        box-shadow: 0 4px 20px rgba(0,200,100,0.15);
    }
    .team-card h3, .team-card h4 { color: #00C9FF; margin: 0 0 6px 0; }
    .team-lead h3 { color: #92FE9D; }
    .team-card p  { color: #7090b0; margin: 2px 0; font-size: 0.9rem; }
    .result-box {
        padding: 14px 18px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.05rem;
        font-weight: bold;
        margin: 6px 0;
        letter-spacing: 0.3px;
    }
    .result-3d {
        background: linear-gradient(135deg, #0a2a0a, #0d1f0d);
        border: 1.5px solid #00C800;
        color: #4dff4d;
        box-shadow: 0 0 12px rgba(0,200,0,0.2);
    }
    .result-2d {
        background: linear-gradient(135deg, #0a0a2a, #0d0d1f);
        border: 1.5px solid #4080ff;
        color: #80a8ff;
        box-shadow: 0 0 12px rgba(0,80,255,0.2);
    }
    .stButton > button {
        background: linear-gradient(90deg, #00C9FF, #0080cc);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(0,150,255,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #92FE9D, #00C9FF);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,200,150,0.4);
        color: #0a0a1a;
    }
    [data-testid="stFileUploader"] {
        background: #0d1f35;
        border: 2px dashed #1a4a7a;
        border-radius: 12px;
        padding: 10px;
    }
    hr { border-color: #1a3a5c; margin: 20px 0; }
    .stAlert { background: #0d1f35; border-radius: 10px; border: 1px solid #1a4a7a; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d1f35, #0a1628);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #1a4a7a;
    }
    [data-testid="stMetricValue"] { color: #00C9FF !important; }
    [data-testid="stMetricDelta"] { color: #92FE9D !important; }
    [data-testid="stRadio"] label { color: #c0d8f0 !important; }
    p, li, span { color: #c0d8f0; }
    h1, h2, h3, h4 { color: #e8f4ff; }
    table { border-collapse: collapse; width: 100%; }
    th { background: #0d1f35; color: #00C9FF; padding: 10px; border: 1px solid #1a4a7a; }
    td { background: #080818; color: #c0d8f0; padding: 8px 10px; border: 1px solid #1a3a5c; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #050510; }
    ::-webkit-scrollbar-thumb { background: #1a4a7a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #00C9FF; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "weights", "best_model.pth"
)
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results"
)


# ── Load model once ───────────────────────────────────────
@st.cache_resource
def load_model():
    load_classifier(WEIGHTS_PATH)
    return True


# ── Process image ─────────────────────────────────────────
def process_image(image_bgr):
    frame_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    depth_map  = get_depth_map(image_bgr)
    detections = detect_objects(image_bgr)

    rgb_out     = image_bgr.copy()
    depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    detection_results = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        obj_label       = det["label"]

        dim_label, conf, reason = classify_roi(
            frame_rgb, depth_map, det["bbox"], obj_label
        )

        color = (0, 200, 80) if dim_label == "3D" else (0, 80, 220)

        # Annotate RGB
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

        # Annotate depth
        cv2.rectangle(depth_color, (x1, y1), (x2, y2), color, 2)
        depth_crop = depth_map[y1:y2, x1:x2].astype(np.float32)
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
                    f"std:{std_d:.0f}  ({reason})",
                    (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 255, 200), 1)

        detection_results.append({
            "object":     obj_label,
            "dimension":  dim_label,
            "confidence": conf,
            "reason":     reason,
            "depth_std":  std_d
        })

    rgb_out     = cv2.cvtColor(rgb_out,     cv2.COLOR_BGR2RGB)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

    return rgb_out, depth_color, detection_results


# ════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Smart Depth Vision")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🖼️ Image Test",
         "📷 Live Webcam", "📊 Results", "👥 Team"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    st.markdown("""
    - **Detector:** YOLOv8n
    - **Depth:** MiDaS Small
    - **Classifier:** MobileNetV2
    - **Accuracy:** 100%
    - **Dataset:** 6,449 images
    - **Epochs:** 20
    """)

    st.markdown("---")
    st.markdown("### 🎨 Legend")
    st.markdown("🟢 **Green** = 3D Object")
    st.markdown("🔵 **Blue** = 2D Object")

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown(
        "[GitHub](https://github.com/sujal-067/Smart-Depth-Vision)"
    )


# ════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════
st.markdown(
    '<h1 class="main-title">🔍 Smart Depth Vision</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-title">Real-Time Monocular Spatial Awareness '
    'Using YOLOv8 and Deep Depth Estimation</p>',
    unsafe_allow_html=True
)
st.markdown("---")


# ════════════════════════════════════════════════════════
#  HOME
# ════════════════════════════════════════════════════════
if page == "🏠 Home":

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>100%</h2>
            <p>Validation Accuracy</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>6,449</h2>
            <p>Training Images</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>1.00</h2>
            <p>F1 Score</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 About This Project")
    st.markdown("""
    **Smart Depth Vision** combines three state-of-the-art models to detect
    objects and classify them as **2D** (flat representations) or
    **3D** (real physical objects):

    1. **YOLOv8** — detects and locates objects with bounding boxes
    2. **MiDaS** — estimates depth from a single RGB image
    3. **Dual-Stream MobileNetV2** — classifies each object using
       RGB + depth features

    > **Core insight:** Real 3D scenes have **high depth variance**
    > while flat 2D images have **low depth variance** in MiDaS depth maps.
    """)

    st.markdown("### 🔧 Pipeline Architecture")
    st.code("""
Input Image / Webcam Frame
        │
        ├──────────────────────┐
        ▼                      ▼
   YOLOv8n                MiDaS Small
 (Detection)            (Depth Map)
        │                      │
        └──────────┬────────────┘
                   ▼
      Depth Variance Analysis
                   │
                   ▼
       2D / 3D Classification
                   │
                   ▼
    Annotated RGB + Depth Output
    """, language="text")

    st.markdown("### 🚀 How to Use")
    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**🖼️ Image Test**\n\n"
            "Upload any image and see real-time\n"
            "2D/3D detection with depth map visualization"
        )
    with col2:
        st.info(
            "**📷 Live Webcam**\n\n"
            "Use your webcam to capture photos\n"
            "for instant 2D/3D analysis"
        )

    st.markdown("### 📊 Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Source | Class | Count |
        |---|---|---|
        | NYU Depth V2 | 3D | 1,449 |
        | COCO Val 2017 | 2D | 5,000 |
        | **Total** | | **6,449** |
        """)
    with col2:
        st.markdown("""
        | Split | Samples |
        |---|---|
        | Training (80%) | 5,160 |
        | Validation (20%) | 1,289 |
        """)


# ════════════════════════════════════════════════════════
#  IMAGE TEST
# ════════════════════════════════════════════════════════
elif page == "🖼️ Image Test":
    st.markdown("## 🖼️ Image Testing")
    st.markdown(
        "Upload an image to test the Smart Depth Vision pipeline.")

    with st.spinner("⏳ Loading model..."):
        load_model()
    st.success("✅ Model loaded successfully!")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload any image to test 2D vs 3D object detection"
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        input_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        st.markdown("### 📥 Input Image")
        st.image(input_rgb,
                 caption="Uploaded Image",
                 use_container_width=True)

        if st.button("🚀 Run Detection", type="primary"):
            with st.spinner("🔍 Running pipeline... Please wait..."):
                rgb_result, depth_result, detections = \
                    process_image(image_bgr)

            st.markdown("### 🎯 Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    rgb_result,
                    caption="RGB + Detection  "
                            "(🟢 Green=3D  |  🔵 Blue=2D)",
                    use_container_width=True
                )
            with col2:
                st.image(
                    depth_result,
                    caption="MiDaS Depth Map + Bounding Boxes",
                    use_container_width=True
                )

            st.markdown("### 📋 Detection Summary")
            if len(detections) == 0:
                st.warning(
                    "⚠️ No objects detected in this image.")
            else:
                for det in detections:
                    css  = ("result-3d"
                            if det['dimension'] == "3D"
                            else "result-2d")
                    icon = ("🟢"
                            if det['dimension'] == "3D"
                            else "🔵")
                    st.markdown(f"""
                    <div class="result-box {css}">
                        {icon} &nbsp;
                        <b>{det['object'].upper()}</b> →
                        {det['dimension']} &nbsp;|&nbsp;
                        Confidence: {det['confidence']:.0%}
                        &nbsp;|&nbsp;
                        {det['reason']}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("### 📈 Statistics")
                total   = len(detections)
                count3d = sum(
                    1 for d in detections
                    if d['dimension'] == '3D')
                count2d = total - count3d
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Objects", total)
                c2.metric("3D Objects 🟢", count3d)
                c3.metric("2D Objects 🔵", count2d)

            st.markdown("### ⬇️ Download Result")
            result_pil = Image.fromarray(rgb_result)
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Annotated Image",
                data=buf.getvalue(),
                file_name="smart_depth_result.png",
                mime="image/png"
            )


# ════════════════════════════════════════════════════════
#  LIVE WEBCAM
# ════════════════════════════════════════════════════════
elif page == "📷 Live Webcam":
    st.markdown("## 📷 Live Webcam Demo")
    st.markdown(
        "Capture a photo using your webcam "
        "for instant 2D/3D analysis.")

    with st.spinner("⏳ Loading model..."):
        load_model()
    st.success("✅ Model loaded!")

    st.info("""
    💡 **Tip:** For smooth real-time video demo, run in terminal:
    ```
    python demo.py
    ```
    """)

    st.markdown("### 📸 Capture Photo")
    img_file = st.camera_input(
        "Click the button below to take a photo")

    if img_file is not None:
        file_bytes = np.asarray(
            bytearray(img_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("🔍 Analyzing..."):
            rgb_result, depth_result, detections = \
                process_image(image_bgr)

        st.markdown("### 🎯 Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb_result,
                     caption="RGB + Detection",
                     use_container_width=True)
        with col2:
            st.image(depth_result,
                     caption="MiDaS Depth Map",
                     use_container_width=True)

        if len(detections) == 0:
            st.warning("⚠️ No objects detected.")
        else:
            st.markdown("### 📋 Detected Objects")
            for det in detections:
                css  = ("result-3d"
                        if det['dimension'] == "3D"
                        else "result-2d")
                icon = ("🟢"
                        if det['dimension'] == "3D"
                        else "🔵")
                st.markdown(f"""
                <div class="result-box {css}">
                    {icon} &nbsp;
                    <b>{det['object'].upper()}</b> →
                    {det['dimension']} &nbsp;|&nbsp;
                    Confidence: {det['confidence']:.0%}
                </div>
                """, unsafe_allow_html=True)

            total   = len(detections)
            count3d = sum(
                1 for d in detections if d['dimension'] == '3D')
            count2d = total - count3d
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", total)
            c2.metric("3D 🟢", count3d)
            c3.metric("2D 🔵", count2d)

    st.markdown("---")
    st.markdown("### 💡 Best Results Tips")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Should detect as 3D 🟢:**
        - Real cup, bottle, mug
        - Person standing/sitting
        - Chair, table, furniture
        - Any real physical object
        """)
    with col2:
        st.markdown("""
        **Should detect as 2D 🔵:**
        - Photo printed on paper
        - Image on phone/laptop screen
        - Poster or magazine page
        - TV screen content
        """)


# ════════════════════════════════════════════════════════
#  RESULTS
# ════════════════════════════════════════════════════════
elif page == "📊 Results":
    st.markdown("## 📊 Evaluation Results")
    st.markdown(
        "Results on **1,289 validation samples** "
        "from full dataset.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  "100.00%", "↑ Perfect")
    col2.metric("F1 Score",  "1.00",    "↑ Perfect")
    col3.metric("Precision", "1.00",    "Both classes")
    col4.metric("Recall",    "1.00",    "Both classes")

    st.markdown("---")
    st.markdown("### 📋 Classification Report")
    report_path = os.path.join(
        RESULTS_DIR, "classification_report.txt")
    if os.path.exists(report_path):
        with open(report_path) as f:
            st.code(f.read(), language="text")
    else:
        st.warning("Run evaluate.py first to generate the report.")

    st.markdown("### 🔲 Confusion Matrix")
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        cm_img = Image.open(cm_path)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(cm_img,
                     caption="Confusion Matrix — 2D vs 3D",
                     use_container_width=True)
    else:
        st.warning("Run evaluate.py to generate confusion matrix.")

    st.markdown("### 📈 Training History")
    hist_path = os.path.join(
        RESULTS_DIR, "training_history.png")
    if os.path.exists(hist_path):
        hist_img = Image.open(hist_path)
        st.image(hist_img,
                 caption="Accuracy and Loss over 20 Epochs",
                 use_container_width=True)
    else:
        st.warning("Run train.py to generate training history.")

    st.markdown("---")
    st.markdown("### 🗃️ Training Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Parameter | Value |
        |---|---|
        | Epochs | 20 |
        | Batch Size | 8 |
        | Learning Rate | 1e-4 |
        | Optimizer | Adam |
        | Device | CPU |
        | Image Size | 128×128 |
        """)
    with col2:
        st.markdown("""
        | Component | Model |
        |---|---|
        | Detector | YOLOv8n |
        | Depth | MiDaS Small |
        | Classifier | MobileNetV2 |
        | Framework | PyTorch 2.10 |
        | Python | 3.12 |
        """)


# ════════════════════════════════════════════════════════
#  TEAM
# ════════════════════════════════════════════════════════
elif page == "👥 Team":
    st.markdown("## 👥 Project Team")
    st.markdown(
        "**Department of Computer Science & Engineering (AI & ML)**")
    st.markdown(
        "Jawaharlal Nehru Govt. Engineering College, "
        "Sundernagar, Mandi, Himachal Pradesh"
    )
    st.markdown("---")

    # Team Lead — Sujal Sharma at top
    st.markdown("""
    <div class="team-card team-lead">
        <h3>⭐ Sujal Sharma — Team Lead</h3>
        <p>Roll No: 23010102065</p>
        <p>B.Tech CSE (Artificial Intelligence & Machine Learning)</p>
        <p>2023 Batch</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Team Members")
    members = [
        ("Abhinav Thakur",  "23010102003"),
        ("Ankit Atri",      "23010102007"),
        ("Ankush",          "23010102008"),
        ("Mritunjay Verma", "23010102038"),
    ]

    col1, col2 = st.columns(2)
    for i, (name, roll) in enumerate(members):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="team-card">
                <h4>👤 {name}</h4>
                <p>Roll No: {roll}</p>
                <p>B.Tech CSE (AI & ML) — 2023 Batch</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="team-card">
        <h4>🎓 Project Supervisor</h4>
        <h3>Er. Rahul Pal Singh</h3>
        <p>Assistant Professor (Computer Engineering)</p>
        <p>JNGEC Sundernagar, Mandi, Himachal Pradesh</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔗 Project Links")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - 📁 **GitHub:** [sujal-067/Smart-Depth-Vision](https://github.com/sujal-067/Smart-Depth-Vision)
        - 📅 **Submitted:** February 2026
        - 🏫 **Institution:** JNGEC Sundernagar
        """)
    with col2:
        st.markdown("""
        - 🐍 **Language:** Python 3.12
        - 🔥 **Framework:** PyTorch 2.10
        - 📦 **Models:** YOLOv8 + MiDaS + MobileNetV2
        """)


# ════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#3a5a7a; font-size:0.85rem;'>"
    "Smart Depth Vision © 2026 &nbsp;|&nbsp; "
    "JNGEC Sundernagar &nbsp;|&nbsp; "
    "B.Tech CSE (AI & ML)</p>",
    unsafe_allow_html=True
)