import streamlit as st
import torch
import time
from pathlib import Path

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(layout="wide")
st.title("Live Webcam Object Detection (YOLOv8)")

st.markdown(
    "Real-time webcam-based object detection using a trained YOLOv8 model. "
    "Displays bounding boxes, FPS, and latency."
)

# =================================================
# TRY LOCAL-ONLY IMPORTS
# =================================================
try:
    import cv2
    from ultralytics import YOLO
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# =================================================
# STREAMLIT CLOUD FALLBACK
# =================================================
if not OPENCV_AVAILABLE:
    st.warning(
        "**Live Webcam Detection is not available on Streamlit Cloud**\n\n"
        "This feature requires:\n"
        "- A physical webcam\n"
        "- OpenCV system libraries (libGL)\n\n"
        "Please run this project **locally** to use webcam detection."
    )

    st.info(
        "**Good news:**\n"
        "- Image Classification \n"
        "- Image Object Detection (YOLOv8) \n"
        "- Model Performance Dashboard \n\n"
        "All other features work perfectly on Streamlit Cloud."
    )
    st.stop()

# =================================================
# DEVICE
# =================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================
# MODEL PATH
# =================================================
BASE_DIR = Path(__file__).parent.parent
YOLO_WEIGHTS_PATH = BASE_DIR / "weights" / "best.pt"

# =================================================
# LOAD MODEL
# =================================================
@st.cache_resource
def load_yolo():
    model = YOLO(YOLO_WEIGHTS_PATH)
    model.to(DEVICE)
    return model

yolo_model = load_yolo()

# =================================================
# SIDEBAR
# =================================================
st.sidebar.header("⚙ Live Detection Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.5, 0.05
)

start_button = st.sidebar.button("Start Webcam")
stop_button = st.sidebar.button("Stop Webcam")

# =================================================
# PLACEHOLDERS
# =================================================
frame_placeholder = st.empty()
metrics_placeholder = st.empty()

# =================================================
# WEBCAM LOOP (LOCAL ONLY)
# =================================================
if start_button:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot access webcam")
    else:
        st.success("Webcam started")

        prev_time = time.time()

        while cap.isOpened():
            if stop_button:
                break

            ret, frame = cap.read()
            if not ret:
                st.warning("⚠ Failed to read frame")
                break

            start_infer = time.time()

            results = yolo_model(
                frame,
                conf=conf_threshold,
                device=DEVICE,
                verbose=False,
            )

            latency_ms = (time.time() - start_infer) * 1000

            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(
                annotated_frame, cv2.COLOR_BGR2RGB
            )

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            frame_placeholder.image(
                annotated_frame,
                channels="RGB",
                use_container_width=True
            )

            metrics_placeholder.markdown(
                f"""
                **Live Performance Metrics**
                - **FPS:** `{fps:.2f}`
                - **Latency:** `{latency_ms:.2f} ms/frame`
                - **Device:** `{DEVICE.upper()}`
                """
            )

        cap.release()
        st.info("Webcam stopped")

else:
    st.info("Click **Start Webcam** to begin live detection")
