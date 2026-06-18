import cv2
import pandas as pd
import numpy as np
import json
import sys

VIDEO_MOTION_THRESHOLD = 15
IMU_ACTIVE_THRESHOLD = 1.2
FPS_SAMPLE_RATE = 2


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_gap = int(fps / FPS_SAMPLE_RATE)

    prev_gray = None
    timeline = []

    frame_count = 0
    timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_gap == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion_score = np.mean(diff)

                if motion_score > VIDEO_MOTION_THRESHOLD:
                    activity = "Active"
                    confidence = min(motion_score / 50, 1.0)
                else:
                    activity = "Static"
                    confidence = 1 - min(motion_score / 50, 1.0)

                timeline.append({
                    "timestamp_ms": int(timestamp_ms),
                    "video_activity": activity,
                    "video_conf": float(confidence)
                })

            prev_gray = gray
            timestamp_ms += 500

        frame_count += 1

    cap.release()
    return timeline


def process_imu(csv_path):
    df = pd.read_csv(csv_path)

    df["magnitude"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)

    df["imu_state"] = df["magnitude"].apply(
        lambda x: "Active" if x > IMU_ACTIVE_THRESHOLD else "Static"
    )

    return df


def fuse(video_timeline, imu_df):
    final_output = []

    imu_times = imu_df["time"].values
    imu_states = imu_df["imu_state"].values

    for entry in video_timeline:
        t = entry["timestamp_ms"]

        idx = np.argmin(np.abs(imu_times - t))

        imu_state = imu_states[idx]
        video_state = entry["video_activity"]
        confidence = entry["video_conf"]

        if imu_state == "Active":
            final_state = "Active"
            confidence = max(confidence, 0.8)
        else:
            final_state = video_state

        final_output.append({
            "timestamp_ms": t,
            "activity": final_state,
            "confidence": round(confidence, 3)
        })

    return final_output


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_pipeline.py <video_path> <imu_csv_path>")
        return

    video_path = sys.argv[1]
    imu_path = sys.argv[2]

    video_data = process_video(video_path)
    imu_data = process_imu(imu_path)
    final_data = fuse(video_data, imu_data)

    with open("timeline.json", "w") as f:
        json.dump(final_data, f, indent=2)

    print("timeline.json created successfully!")


if __name__ == "__main__":
    main()
