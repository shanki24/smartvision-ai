import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(layout="wide")
st.title("📊 Model Performance Dashboard")

st.markdown(
    "This dashboard provides a **comparative performance analysis** of all "
    "trained models, including **accuracy metrics, inference speed, confusion "
    "matrices, and class-wise performance breakdown**."
)

# =================================================
# PATH CONFIG
# =================================================
BASE_DIR = Path(__file__).parent.parent
EVAL_DIR = BASE_DIR / "evaluation"

MODELS = {
    "VGG16": "vgg16",
    "ResNet50": "resnet50",
    "MobileNetV2": "mobilenetv2",
    "EfficientNetB0": "efficientnetb0",
}

# =================================================
# LOAD METRICS
# =================================================
metrics_data = {}

for name, key in MODELS.items():
    metrics_path = EVAL_DIR / f"{key}_metrics.csv"
    if metrics_path.exists():
        metrics_data[name] = pd.read_csv(metrics_path)
    else:
        st.warning(f"Metrics file not found: {metrics_path}")

# =================================================
# MODEL COMPARISON – ACCURACY
# =================================================
st.header("📈 Accuracy Comparison")

if metrics_data:
    acc_df = pd.DataFrame({
        model: df["accuracy"].values[0]
        for model, df in metrics_data.items()
    }, index=["Accuracy"]).T

    st.bar_chart(acc_df)

# =================================================
# INFERENCE SPEED COMPARISON
# =================================================
st.header("⚡ Inference Speed (ms)")

speed_df = pd.DataFrame({
    model: df["inference_time_ms"].values[0]
    for model, df in metrics_data.items()
}, index=["Inference Time (ms)"]).T

st.bar_chart(speed_df)

# =================================================
# MODEL SIZE COMPARISON
# =================================================
st.header("💾 Model Size (MB)")

size_df = pd.DataFrame({
    model: df["model_size_mb"].values[0]
    for model, df in metrics_data.items()
}, index=["Model Size (MB)"]).T

st.bar_chart(size_df)

# =================================================
# CONFUSION MATRICES
# =================================================
st.header("🧮 Confusion Matrices")

cols = st.columns(2)

for i, (name, key) in enumerate(MODELS.items()):
    cm_path = EVAL_DIR / f"{key}_confusion.npy"

    if cm_path.exists():
        cm = np.load(cm_path)

        with cols[i % 2]:
            st.subheader(name)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(cm, cmap="Blues")
            ax.set_title(f"{name} Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()

            st.pyplot(fig)

# =================================================
# CLASS-WISE PERFORMANCE
# =================================================
st.header("📌 Class-wise Performance Breakdown")

selected_model = st.selectbox(
    "Select Model",
    list(MODELS.keys())
)

if selected_model in metrics_data:
    st.dataframe(metrics_data[selected_model])
else:
    st.info("Select a model to view class-wise metrics.")
