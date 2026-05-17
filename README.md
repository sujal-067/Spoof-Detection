# 🔐 Spoof Detector – Face Anti-Spoofing & Authentication System

Spoof Detector is an advanced **AI-powered authentication and anti-spoofing system** that combines **face recognition, liveness detection, and depth-aware analysis** to provide secure user verification.

The system allows users to register, enroll facial data, and authenticate using either **credentials or facial recognition**, while actively detecting spoof attempts.

---

## 🚀 Key Features

### 🔑 Authentication System

* Email & Password-based Login
* Face Recognition Login
* Secure user registration

### 🧠 Face Enrollment (Multi-Angle)

Users must upload **4 facial images** during enrollment:

1. 📸 Front-facing
2. ➡️ Facing right
3. ⬅️ Facing left
4. ⬆️ Head tilted upward

⚠️ **Important:**

* No sunglasses or spectacles allowed
* Clear face visibility required

---

### 🛡️ Anti-Spoofing System

* Detects fake attempts (photo, screen replay, etc.)
* Uses depth + facial cues for verification
* Enhances real-user validation

---

### 🎥 Live Analysis Mode

* Real-time webcam-based verification
* Continuous spoof detection
* Instant feedback

---

### 📷 Capture & Verify

* Automatically captures 4 images from live video
* Matches with enrolled data
* Fully automated verification pipeline

---

### 📊 User Dashboard Features

* View authentication logs
* Re-enroll facial data
* Monitor activity

---

## 🧠 How It Works

1. User registers using email & password
2. User uploads 4 structured facial images
3. System processes and stores embeddings
4. During login:

   * Option 1: Credentials
   * Option 2: Face Recognition
5. Anti-spoofing model verifies authenticity
6. Access granted only if verified

---

## 🏗️ Project Structure

```id="projstruct1"
Smart-Depth-Vision/
│── app/
│   ├── main.py
│   ├── auth/
│   ├── spoof_detection/
│   ├── face_recognition/
│   ├── depth/
│   ├── routes/
│── database/
│── models/
│── utils/
│── static/
│── templates/
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

```bash id="installcmd1"
git clone https://github.com/Ankit-13-dev/Smart-Depth-Vision.git
cd Smart-Depth-Vision

pip install -r requirements.txt
```

---

## ▶️ Running the Project

```bash id="runcmd1"
uvicorn app.main:app --reload
```

Then open in browser:

```
http://localhost:8000
```

---

## 🧪 Use Cases

* 🔐 Secure login systems
* 🏦 Banking & fintech authentication
* 🧑‍💻 Attendance systems
* 📱 Identity verification platforms
* 🛂 Access control systems

---

## 🛠️ Tech Stack

* Python 🐍
* FastAPI ⚡
* OpenCV 👁️
* Face Recognition Models
* Depth Estimation Models
* Anti-Spoofing Models
* YOLOv8 (optional modules)

---

## 🔮 Future Enhancements

* Mobile app integration 📱
* 3D face reconstruction
* IR / depth sensor integration
* Cloud deployment (AWS/GCP)
* Multi-user scalability

---

## 📸 Demo

(Add screenshots / GIFs here for better visibility)

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork this repo and submit pull requests.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Ankit Atri**
GitHub: https://github.com/Ankit-13-dev
