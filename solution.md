# WigglyWoosh Technical Challenge Solution

## Overview

This solution implements a CPU-efficient pipeline to classify a dog's activity (Active or Static) using both video and IMU sensor data.

---

## 1. Video Classification

The video is processed using OpenCV:

- Frames are converted to grayscale
- Frame differencing is applied
- Mean pixel difference is used as motion score

### Rule:
- Motion score > threshold → Active
- Otherwise → Static

This method is lightweight and runs efficiently on CPU.

---

## 2. IMU Processing

Acceleration magnitude is calculated using:

magnitude = sqrt(ax² + ay² + az²)

### Rule:
- Magnitude > threshold → Active
- Otherwise → Static

---

## 3. Sensor Fusion

To improve reliability:

- If IMU detects "Active", it overrides video prediction
- Otherwise, video prediction is used

This helps handle cases like:
- Camera obstruction
- Low motion visibility

---

## 4. Output

- Output is generated at 2 Hz (every 500 ms)
- Saved as `timeline.json`

Each entry contains:
- timestamp (ms)
- activity (Active/Static)
- confidence score

---

## Conclusion

The solution is:
- Efficient (CPU-friendly)
- Robust (sensor fusion)
- Simple and explainable
