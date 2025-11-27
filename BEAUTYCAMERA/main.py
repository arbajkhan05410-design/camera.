import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os

st.set_page_config(page_title="Beauty Camera", layout="wide")

st.title("📸 Beauty Camera with Filters, Photo Capture & Video Recording")

# Create output folder
if not os.path.exists("output"):
    os.makedirs("output")

filters = [
    "None",
    "Blur",
    "Brightness+",
    "Contrast+",
    "Gray",
    "Sepia",
    "Cartoon",
]

selected_filter = st.selectbox("Choose Filter", filters)

start = st.button("Start Camera")
stop = st.button("Stop Camera")

capture_photo = st.button("Capture Photo")
record_video = st.checkbox("Record Video")

frame_placeholder = st.empty()

video_frames = []

def apply_filter(frame, f):
    if f == "Blur":
        return cv2.GaussianBlur(frame, (25, 25), 0)

    elif f == "Brightness+":
        return cv2.convertScaleAbs(frame, alpha=1, beta=40)

    elif f == "Contrast+":
        return cv2.convertScaleAbs(frame, alpha=1.5, beta=0)

    elif f == "Gray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    elif f == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(frame, kernel)

    elif f == "Cartoon":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(frame, 9, 250, 250)
        return cv2.bitwise_and(color, color, mask=edges)

    return frame


if start:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not found!")
            break

        frame = cv2.flip(frame, 1)

        filtered = apply_filter(frame, selected_filter)

        if selected_filter == "Gray":
            frame_placeholder.image(filtered, channels="GRAY")
        else:
            frame_placeholder.image(filtered, channels="BGR")

        if capture_photo:
            filename = f"output/photo_{int(time.time())}.jpg"
            cv2.imwrite(filename, filtered)
            st.success(f"📷 Photo Saved: {filename}")

        if record_video:
            video_frames.append(filtered)

        if stop:
            break

    cap.release()

# Save video
if record_video and len(video_frames) > 0:
    h, w = video_frames[0].shape[:2]
    filename = f"output/video_{int(time.time())}.mp4"

    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))

    for frame in video_frames:
        if len(frame.shape) == 2:  # convert gray to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(frame)

    writer.release()
    st.success(f"🎥 Video Saved: {filename}")
    st.video(filename)