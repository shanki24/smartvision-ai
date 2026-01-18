import streamlit as st

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(layout="wide")
st.title("ℹ️ About This Project")

# =================================================
# PROJECT OVERVIEW
# =================================================
st.header("📌 Project Overview")

st.markdown(
    """
    **SmartVision AI** is an end-to-end computer vision platform that demonstrates
    **image classification**, **object detection**, **model evaluation**, and
    **real-time inference** using state-of-the-art deep learning models.

    The project is designed as a **complete academic + industry-ready pipeline**,
    starting from dataset preparation and model training, all the way to
    interactive deployment using **Streamlit**.
    """
)

# =================================================
# DATASET INFORMATION
# =================================================
st.header("📂 Dataset Information")

st.markdown(
    """
    - **Dataset Type:** Custom curated dataset inspired by COCO-style structure  
    - **Tasks Supported:**
        - Image Classification (Single object per image)
        - Object Detection (Bounding boxes + class labels)
    - **Image Format:** JPG / PNG
    - **Input Resolution:** 224×224 (classification), dynamic resizing (YOLO)
    - **Number of Classes:** Defined during training (classification & detection)
    """
)

# =================================================
# MODEL ARCHITECTURES
# =================================================
st.header("🧠 Model Architectures Used")

st.markdown(
    """
    ### 🔹 Image Classification Models
    - **VGG16**
        - Deep CNN with stacked convolution layers
        - Strong baseline for visual feature extraction
    - **ResNet50**
        - Residual connections for deeper networks
        - Improved gradient flow and accuracy
    - **MobileNetV2**
        - Lightweight and optimized for speed
        - Ideal for edge and real-time applications
    - **EfficientNet-B0**
        - Compound scaling of depth, width, and resolution
        - Best accuracy-to-efficiency ratio

    ### 🔹 Object Detection Model
    - **YOLOv8 (Ultralytics)**
        - Single-stage real-time object detector
        - High FPS with strong detection accuracy
        - Used for both image and live webcam detection
    """
)

# =================================================
# TECHNICAL STACK
# =================================================
st.header("🛠️ Technical Stack")

st.markdown(
    """
    **Programming & Frameworks**
    - Python 3.11
    - PyTorch
    - Torchvision
    - Ultralytics YOLOv8

    **Data & Visualization**
    - NumPy
    - Pandas
    - Matplotlib
    - OpenCV

    **Deployment**
    - Streamlit (Multi-page application)
    - CUDA (GPU acceleration)
    - Virtual Environment (venv)

    **Development Tools**
    - Jupyter Notebook
    - VS Code
    """
)

# =================================================
# PROJECT WORKFLOW
# =================================================
st.header("🔄 Project Workflow")

st.markdown(
    """
    1. **Dataset Preparation**
        - Folder structuring
        - Train / validation split
    2. **Model Training**
        - Transfer learning with CNN backbones
        - Validation-based model checkpointing
    3. **Evaluation**
        - Accuracy
        - Inference time
        - Model size
        - Confusion matrices
    4. **Deployment**
        - Streamlit UI
        - Image upload inference
        - YOLO object detection
        - Live webcam detection
    """
)

# =================================================
# KEY FEATURES
# =================================================
st.header("✨ Key Features")

st.markdown(
    """
    - 📸 Image Classification with multiple CNNs
    - 🎯 Object Detection with YOLOv8
    - 📊 Model Performance Dashboard
    - 🎥 Live Webcam Object Detection
    - ⚡ GPU-accelerated inference
    - 🧩 Modular & scalable architecture
    """
)

# =================================================
# DEVELOPER INFORMATION
# =================================================
st.header("👨‍💻 Developer Information")

st.markdown(
    """
    **Developer:** Shashank Shandilya
    **Domain:** Artificial Intelligence & Machine Learning Student
    **Focus Areas:**
    - Computer Vision
    - Deep Learning
    - Model Optimization
    - AI Application Deployment

    This project was developed as a **full-stack AI system** showcasing
    both **theoretical understanding** and **practical deployment skills**.
    """
)

# =================================================
# FOOTER
# =================================================
st.markdown("---")
st.markdown(
    "🚀 *SmartVision AI — From Models to Real-Time Vision*"
)
