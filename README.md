# WiggyWoosh
🐶 WigglyWoosh Technical Challenge – Dog Activity Detection

📌 Overview

This project implements a lightweight Python pipeline to classify a dog’s activity (Active or Static) by combining:

* 🎥 Video data (dog_video.mp4)
* 📊 IMU sensor data (collar_imu.csv)

The system is optimized to run efficiently on a standard CPU and uses sensor fusion to improve prediction accuracy.

⸻

⚙️ Approach

1. Video-Based Classification

* Frames are processed using OpenCV
* Motion is detected using frame differencing
* Mean pixel difference is used as a motion score

Rule:

* High motion → Active
* Low motion → Static

⸻

2. IMU-Based Classification

* Acceleration magnitude is calculated:
    magnitude = sqrt(ax² + ay² + az²)

Rule:

* High magnitude → Active
* Low magnitude → Static

⸻

3. Sensor Fusion

* IMU is used to correct video errors
* If IMU detects Active, it overrides video prediction
* Otherwise, video prediction is retained

⸻

📤 Output

The pipeline generates:

📁 timeline.json

* Sampled at 2 Hz (every 500 ms)
* Format:

{
  "timestamp_ms": 1000,
  "activity": "Active",
  "confidence": 0.87
}

⸻

▶️ How to Run

Install dependencies:

pip install opencv-python pandas numpy

Run the pipeline:

python run_pipeline.py dog_video.mp4 collar_imu.csv

⸻

📦 Repository Structure

.
├── run_pipeline.py
├── solution.md
├── README.md

⸻

🚀 Key Features

* ✅ CPU-efficient (no heavy deep learning models)
* ✅ Real-time capable
* ✅ Robust using sensor fusion
* ✅ Simple and explainable logic

⸻

📈 Possible Improvements

* Use lightweight deep learning models (e.g., MobileNet)
* Apply smoothing filters on IMU signals
* Add temporal consistency using sliding windows

⸻

