import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(layout="wide")
st.title("🖼 Image Classification")

st.markdown(
    "Upload a **single-object image** and view predictions from "
    "**VGG16, ResNet50, MobileNetV2, and EfficientNetB0**."
)

# =================================================
# GLOBAL CONFIG
# =================================================
DEVICE = torch.device("cpu")
NUM_CLASSES = 29

MODEL_PATHS = {
    "VGG16": r"C:\Users\Sujal\OneDrive\Desktop\coco_project\vgg16_best.pth",
    "ResNet50": r"C:\Users\Sujal\OneDrive\Desktop\coco_project\resnet50_best.pth",
    "MobileNetV2": r"C:\Users\Sujal\OneDrive\Desktop\coco_project\mobilenetv2_best.pth",
    "EfficientNetB0": r"C:\Users\Sujal\OneDrive\Desktop\coco_project\efficientnetb0_best.pth",
}

# =================================================
# IMAGE TRANSFORM (MATCH TRAINING)
# =================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

# =================================================
# LOAD MODELS (MATCH IPYNB EXACTLY)
# =================================================
@st.cache_resource
def load_models():
    models_dict = {}

    # ------------------ VGG16 ------------------
    vgg = models.vgg16(weights=None)
    vgg.classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(256, NUM_CLASSES),
    )
    vgg.load_state_dict(torch.load(MODEL_PATHS["VGG16"], map_location=DEVICE))
    vgg.eval()
    models_dict["VGG16"] = vgg

    # ------------------ ResNet50 ------------------
    resnet = models.resnet50(weights=None)
    resnet.fc = torch.nn.Sequential(
        torch.nn.Linear(resnet.fc.in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, NUM_CLASSES),
    )
    resnet.load_state_dict(torch.load(MODEL_PATHS["ResNet50"], map_location=DEVICE))
    resnet.eval()
    models_dict["ResNet50"] = resnet

    # ------------------ MobileNetV2 ------------------
    mobilenet = models.mobilenet_v2(weights=None)
    mobilenet.classifier[1] = torch.nn.Linear(
        mobilenet.classifier[1].in_features,
        NUM_CLASSES
    )
    mobilenet.load_state_dict(
        torch.load(MODEL_PATHS["MobileNetV2"], map_location=DEVICE)
    )
    mobilenet.eval()
    models_dict["MobileNetV2"] = mobilenet

    # ------------------ EfficientNetB0 (FIXED) ------------------
    effnet = models.efficientnet_b0(weights=None)
    effnet.classifier = torch.nn.Sequential(
        torch.nn.BatchNorm1d(1280),
        torch.nn.Linear(1280, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, NUM_CLASSES),
    )
    effnet.load_state_dict(
        torch.load(MODEL_PATHS["EfficientNetB0"], map_location=DEVICE)
    )
    effnet.eval()
    models_dict["EfficientNetB0"] = effnet

    return models_dict


models_dict = load_models()

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
    st.image(image, width="stretch")

    input_tensor = transform(image).unsqueeze(0)

    st.subheader("📊 Model Predictions (Top-5)")
    cols = st.columns(4)

    for col, (model_name, model) in zip(cols, models_dict.items()):
        with col:
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1)

            top5_prob, top5_idx = torch.topk(probs, 5)

            st.markdown(f"### 🔹 {model_name}")
            for i in range(5):
                st.write(
                    f"Class {top5_idx[0][i].item()} → "
                    f"**{top5_prob[0][i].item() * 100:.2f}%**"
                )
else:
    st.info("👆 Upload an image to start classification.")
