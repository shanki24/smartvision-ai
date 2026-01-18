import streamlit as st
from PIL import Image
from pathlib import Path

st.title("🏠 SmartVision AI – Home")

# ---------------- PROJECT OVERVIEW ----------------
st.header("📌 Project Overview")

st.markdown("""
**SmartVision AI** is an end-to-end **computer vision system** that performs:

- 🔍 **Multi-object detection** using YOLOv8  
- 🧠 **Single-object classification** using CNNs  
- ⚡ Optimized inference with **GPU, FP16 & INT8 quantization**
- 🌐 Deployment-ready **Streamlit web application**

The system is trained on a **25-class curated subset of the COCO dataset**
and designed for **real-world use cases** such as smart cities, retail,
security, healthcare, and automation.
""")

# ---------------- KEY FEATURES ----------------
st.header("🚀 Key Features")

st.markdown("""
- ✅ YOLOv8 multi-object detection with bounding boxes  
- ✅ CNN classification (VGG16, ResNet50, MobileNetV2, EfficientNetB0)  
- ✅ Model comparison & performance analytics  
- ✅ GPU-accelerated inference (CUDA enabled)  
- ✅ Cloud-ready optimization (low memory footprint)  
""")

# ---------------- HOW TO USE ----------------
st.header("🛠 How to Use the Application")

st.markdown("""
1. Navigate to **Image Classification** to classify a single object image  
2. Navigate to **Object Detection** to detect multiple objects in an image  
3. Adjust confidence thresholds for detection  
4. View detailed metrics in **Model Performance**  
""")

# ---------------- SAMPLE DEMO IMAGES ----------------
st.header("🖼 Sample Demo Images")

demo_dir = Path(__file__).parent.parent / "assets" / "demo_images"

if demo_dir.exists():
    cols = st.columns(3)
    images = list(demo_dir.glob("*.*"))[:3]

    for col, img_path in zip(cols, images):
        with col:
            img = Image.open(img_path)
            st.image(img, caption=img_path.name, use_container_width=True)
else:
    st.warning("Demo images not found. Add images to assets/demo_images/")
