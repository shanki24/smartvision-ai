import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(layout="wide")
st.title("🎯 Object Detection (YOLOv8)")

st.markdown(
    "Upload an image to perform **multi-object detection** using a trained "
    "**YOLOv8 model**. Detected objects are shown with **bounding boxes, "
    "class labels, and confidence scores**."
)

# =================================================
# GLOBAL CONFIG
# =================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_WEIGHTS_PATH = (
    r"C:\Users\Sujal\OneDrive\Desktop\coco_project\yolo_training\weights\best.pt"
)

# =================================================
# LOAD YOLO MODEL (CACHED)
# =================================================
@st.cache_resource
def load_yolo():
    model = YOLO(YOLO_WEIGHTS_PATH)
    model.to(DEVICE)
    return model


yolo_model = load_yolo()

# =================================================
# CONFIDENCE THRESHOLD
# =================================================
st.sidebar.header("⚙ Detection Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

# =================================================
# IMAGE UPLOAD
# =================================================
uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📸 Uploaded Image")
    st.image(image, width=600)

    # =================================================
    # YOLO INFERENCE
    # =================================================
    with st.spinner("Running YOLO object detection..."):
        results = yolo_model(
            image,
            conf=conf_threshold,
            device=DEVICE,
        )

    result = results[0]

    # =================================================
    # DISPLAY IMAGE WITH BOUNDING BOXES
    # =================================================
    st.subheader("🧠 Detection Results")

    annotated_img = result.plot()  # numpy array (BGR)
    annotated_img = annotated_img[:, :, ::-1]  # BGR → RGB

    st.image(annotated_img, width=600)

    # =================================================
    # DISPLAY DETECTION DETAILS
    # =================================================
    if result.boxes is not None and len(result.boxes) > 0:
        st.subheader("📋 Detected Objects")

        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            label = result.names[cls_id]

            st.write(
                f"**{i+1}. {label}**  |  Confidence: **{conf:.2f}**"
            )
    else:
        st.warning("No objects detected at the selected confidence threshold.")

else:
    st.info("👆 Upload an image to start object detection.")
