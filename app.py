import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
BANNER_PATH = BASE_DIR / "assets" / "demo_images" / "banner.png"

st.image(BANNER_PATH, use_container_width=True)

st.sidebar.title("📌 SmartVision AI")
st.sidebar.markdown("""
An Intelligent Multi-Class  
Object Recognition System
""")

st.sidebar.info(
    "Use the navigation menu to explore "
    "classification, detection, and model insights."
)

st.title("SmartVision AI")
st.subheader("Intelligent Multi-Class Object Recognition System")

st.markdown("""
Welcome to **SmartVision AI**, a production-ready computer vision platform  
built using **CNN-based classification models** and **YOLOv8 object detection**.

👉 Navigate using the **sidebar** to explore different modules.
""")
