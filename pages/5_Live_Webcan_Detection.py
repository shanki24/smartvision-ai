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
# IMPORTS (CLOUD COMPATIBLE)
# =================================================
try:
    import cv2
    import av
    from ultralytics import YOLO
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    OPENCV_AVAILABLE = True
except Exception as e:
    st.error(f"Import error: {e}")
    OPENCV_AVAILABLE = False

# =================================================
# STREAMLIT CLOUD FALLBACK
# =================================================
if not OPENCV_AVAILABLE:
    st.warning(
        "**Live Webcam Detection is not available on Streamlit Cloud**\n\n"
        "This feature requires:\n"
        "- OpenCV system libraries (libGL)\n\n"
        "Please ensure dependencies are installed."
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

# =================================================
# PLACEHOLDERS
# =================================================
metrics_placeholder = st.empty()

# =================================================
# VIDEO PROCESSOR (WEBRTC)
# =================================================
class YOLOVideoProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        start_infer = time.time()

        results = yolo_model(
            img,
            conf=conf_threshold,
            device=DEVICE,
            verbose=False,
        )

        latency_ms = (time.time() - start_infer) * 1000

        annotated_frame = results[0].plot()

        metrics_placeholder.markdown(
            f"""
            **Live Performance Metrics**
            - **Latency:** `{latency_ms:.2f} ms/frame`
            - **Device:** `{DEVICE.upper()}`
            """
        )

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )

# =================================================
# WEBCAM STREAM (CLOUD COMPATIBLE)
# =================================================
webrtc_streamer(
    key="yolo-live",
    video_processor_factory=YOLOVideoProcessor,
)

st.info("Allow camera access in your browser to start live detection.")
