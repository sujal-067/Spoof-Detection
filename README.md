# Smart Depth Vision
### Real-Time Monocular Spatial Awareness Using YOLOv8 and Deep Depth Estimation

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-orange?style=flat-square&logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green?style=flat-square)
![MiDaS](https://img.shields.io/badge/Depth-MiDaS-purple?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-blue?style=flat-square)

---

## Overview

**Smart Depth Vision** is a deep learning system that detects objects in real time and classifies them as either **2D** (flat representations such as photos, posters, or screen content) or **3D** (real physical objects in the environment).

Most existing object detection systems can identify *what* an object is, but cannot determine *whether it is physically real or just a flat image*. This limitation causes problems in:

- **Robotics** — a robot must not attempt to pick up a photograph of an object
- **Augmented Reality** — virtual content must align with real 3D geometry
- **Smart Surveillance** — systems must distinguish real people from printed photographs

Our system solves this by combining **object detection**, **monocular depth estimation**, and **depth-aware classification** into a single unified pipeline.

---

## Demo

| RGB + Detection | MiDaS Depth Map |
|---|---|
| Objects detected with 2D/3D labels and confidence | Depth heatmap with bounding boxes and depth statistics |

- 🟢 **Green box** = Real 3D object
- 🔵 **Blue box** = Flat 2D representation

---

## How It Works

```
Input Image / Webcam Frame
        │
        ├──────────────────────┐
        ▼                      ▼
   YOLOv8n                  MiDaS Small
 (Object Detection)      (Depth Estimation)
        │                      │
        └──────────┬───────────┘
                   ▼
          ROI Depth Extractor
       (crop bounding box regions)
                   │
                   ▼
        Depth Variance Analysis
      + Semantic Class Override
                   │
                   ▼
         2D / 3D Classification
                   │
                   ▼
        Annotated Output Frame
     (RGB + Depth Map side by side)
```

### Three-Stage Pipeline

1. **YOLOv8** detects and locates objects with bounding boxes
2. **MiDaS** generates a monocular depth map from the single RGB image
3. **Classifier** analyzes depth variance within each bounding box:
   - High depth variance → **3D object** (real world)
   - Low depth variance → **2D object** (flat representation)

---

## Model Architecture

```
RGB Image  ──► MobileNetV2 encoder ──► 1280-d features ──┐
                                                           ├──► concat(2560) ──► FC(512) ──► FC(128) ──► 2 classes
Depth Map  ──► MobileNetV2 encoder ──► 1280-d features ──┘
```

- **Stream 1** — processes RGB image for visual and texture features
- **Stream 2** — processes depth map for geometric and spatial features
- **Fusion head** — combines both streams for the final 2D/3D decision

---

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | **100.00%** |
| F1 Score (2D) | **1.00** |
| F1 Score (3D) | **1.00** |
| Precision (2D) | 1.00 |
| Recall (2D) | 1.00 |
| Precision (3D) | 1.00 |
| Recall (3D) | 1.00 |
| Validation Samples | 1289 |
| Total Dataset Size | 6449 images |

### Training History

| Epoch | Train Accuracy | Val Accuracy |
|---|---|---|
| 1 | 96.8% | 99.9% |
| 5 | 99.9% | 100.0% |
| 10 | 100.0% | 100.0% |
| 20 | 100.0% | 100.0% |

---

## Dataset

| Source | Class | Count | Description |
|---|---|---|---|
| NYU Depth V2 | 3D | 1449 | Real indoor RGB-D scenes |
| COCO Val 2017 | 2D | 5000 | Flat photographs |
| **Total** | | **6449** | **80% train / 20% val** |

**Key insight:** MiDaS depth maps of real 3D scenes have significantly higher depth variance (std ~45) compared to flat 2D photographs (std ~12). This difference is the core signal our model learns.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Object Detection | YOLOv8n (Ultralytics) | Locate objects in frame |
| Depth Estimation | MiDaS Small | Generate depth map |
| Classifier | Dual-Stream MobileNetV2 | 2D vs 3D classification |
| Framework | PyTorch 2.10 | Model training and inference |
| Computer Vision | OpenCV 4.11 | Image processing |

---

## Project Structure

```
Smart_Depth_Vision/
│
│   train.py                  ← Training script
│   evaluate.py               ← Evaluation + metrics
│   demo.py                   ← Image / webcam demo
│   requirements.txt          ← Python dependencies
│   Smart_Depth_Vision.ipynb  ← Jupyter notebook
│
├───modules/
│       dataset.py            ← Data loader (NYU + COCO)
│       classifier.py         ← Dual-stream MobileNetV2
│       depth.py              ← MiDaS depth estimation
│       detector.py           ← YOLOv8 object detection
│       pipeline.py           ← Full inference pipeline
│
├───data/
│   ├───extracted/
│   │   ├───images/           ← NYU RGB images (not in repo)
│   │   └───depths/           ← NYU depth maps (not in repo)
│   └───2d_images/val2017/    ← COCO images (not in repo)
│
├───weights/
│       history.json          ← Training history
│       best_model.pth        ← Saved weights (not in repo)
│
└───results/
        confusion_matrix.png
        training_history.png
        classification_report.txt
```

---

## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/sujal-067/Smart-Depth-Vision.git
cd Smart-Depth-Vision
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download datasets

**NYU Depth V2:**
- Download from: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- Place RGB images in `data/extracted/images/`
- Place depth maps in `data/extracted/depths/`

**COCO Val 2017:**
```bash
# Download (1 GB)
wget http://images.cocodataset.org/zips/val2017.zip
# Extract to data/2d_images/val2017/
```

---

## Usage

### Train the model
```bash
python train.py
```

### Evaluate on validation set
```bash
python evaluate.py
```

### Run demo on a single image
```bash
python demo.py path/to/image.png
```

### Run live webcam demo
```bash
python demo.py
```

**Controls during demo:**
- Press `s` — save screenshot
- Press `q` — quit

---

## Semantic Class Override

To handle known edge cases, the pipeline uses semantic rules:

| Rule | Classes | Result |
|---|---|---|
| Screen detection | tv, laptop, phone, monitor | Always 2D |
| Flat surfaces | book, poster, magazine, painting | Always 2D |
| Known 3D objects | person, chair, cup, car, etc. | 3D (depth verified) |
| Unknown objects | anything else | Depth variance analysis |

---

## Limitations

1. **Digital screens** — physical monitor bodies are 3D objects; content displayed on them is 2D. Addressed via semantic screen class override.
2. **Embedded images** — images printed on objects (e.g. butterfly on book cover) cannot be detected as 2D sub-regions without a second-stage classifier.
3. **Occluded objects** — partially hidden objects may be missed by YOLOv8.
4. **Textureless surfaces** — blank walls and floors provide insufficient depth cues for MiDaS.

---

## Future Work

- Deploy on edge devices (Raspberry Pi, NVIDIA Jetson Nano)
- Implement second-stage embedded image detection
- Add GPU support for real-time performance
- Extend to multi-class depth-based scene understanding
- Train on more diverse 2D scenarios (billboards, paintings, digital signage)

---

## Team

| Name | Roll No |
|---|---|
| Abhinav Thakur | 23010102003 |
| Ankit Atri | 23010102007 |
| Ankush | 23010102008 |
| Mritunjay Verma | 23010102038 |
| Sujal Sharma | 23010102065 |

**Supervisor:** Er. Rahul Pal Singh, Assistant Professor (Computer Engineering)  
**Institution:** Jawaharlal Nehru Govt. Engineering College, Sundernagar, Mandi, HP  
**Submitted:** February 2026

---

## References

1. Ranftl et al., "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer", IEEE TPAMI, 2022
2. Jocher, Chaurasia, Qiu, "YOLOv8 by Ultralytics", 2023
3. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR, 2021
4. Song et al., "SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite", CVPR, 2015
5. Gupta et al., "Learning Rich Features from RGB-D Images for Object Detection and Segmentation", ECCV, 2014
6. Simonyan and Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", ICLR, 2015

---

*This project is submitted as part of B.Tech coursework in Computer Science & Engineering (Artificial Intelligence & Machine Learning), 2026.*
