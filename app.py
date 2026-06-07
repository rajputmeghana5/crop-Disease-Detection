import streamlit as st


st.set_page_config(page_title="Crop Disease Dashboard", layout="wide")

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd

# =========================
# CONFIG
# =========================
MODEL_PATH = "model/best_model.h5"
IMG_SIZE = (128, 128)


import os
import requests

def download_model():
    model_path = "model/best_model.h5"

    # create folder if not exists
    os.makedirs("model", exist_ok=True)

    # if model not already downloaded
    if not os.path.exists(model_path):
        url = "https://github.com/gayatrishirsath73-cyber/Crop-Disease-Detection-System/releases/download/Model/best_model.h5"

        print("Downloading model...")

        response = requests.get(url, stream=True)

        # check if request successful
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:   # avoid empty chunks
                        f.write(chunk)
            print("Model downloaded successfully")
        else:
            raise Exception(f"Download failed! Status code: {response.status_code}")

# call function before loading model
download_model()
# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# LOAD CLASS NAMES
# =========================
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# =========================
# DISEASE INFO
# =========================
DISEASE_INFO = {
    "healthy": {
        "desc": "Plant is healthy.",
        "treatment": "Maintain watering, sunlight, and soil nutrients."
    },
    "Bacterial_spot": {
        "desc": "Bacterial disease causing spots.",
        "treatment": "Use copper fungicide and remove infected leaves."
    },
    "Early_blight": {
        "desc": "Fungal disease causing brown spots.",
        "treatment": "Apply fungicide and improve airflow."
    },
    "Late_blight": {
        "desc": "Serious fungal disease.",
        "treatment": "Avoid wet leaves and use resistant seeds."
    },
    "Leaf_Mold": {
        "desc": "Fungal infection in humid conditions.",
        "treatment": "Reduce humidity and apply fungicide."
    },
    "Spider_mites": {
        "desc": "Pest infestation damaging leaves.",
        "treatment": "Use neem oil spray."
    },
    "YellowLeaf": {
        "desc": "Virus causing yellow leaves.",
        "treatment": "Control whiteflies."
    },
    "mosaic_virus": {
        "desc": "Virus causing mosaic pattern.",
        "treatment": "Remove infected plants."
    }
}

# =========================
# FIXED PREDICTION FUNCTION
# =========================
def predict_image(img):
    #  FIX: RGBA → RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)[0]
    class_index = np.argmax(predictions)
    confidence = predictions[class_index]

    return class_index, confidence, predictions

# =========================
# GET DISEASE INFO
# =========================
def get_info(disease):
    for key in DISEASE_INFO:
        if key.lower() in disease.lower():
            return DISEASE_INFO[key]
    return {"desc": "No info available", "treatment": "Monitor plant."}

# =========================
# REPORT GENERATION
# =========================
def generate_report(disease, confidence):
    data = {
        "Prediction": [disease],
        "Confidence": [confidence]
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')

# =========================
# PREMIUM CSS UI
# =========================
st.markdown("""
<style>

/* ===== GLOBAL ===== */
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
    background-color: #f4f7fb;
}

/* ===== MAIN CONTAINER ===== */
.main {
    background-color: #f4f7fb;
    padding: 10px;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    padding: 20px;
    border-right: 1px solid rgba(255,255,255,0.1);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar buttons */
.stSidebar button {
    background-color: transparent;
    border-radius: 10px;
    transition: 0.3s;
}

.stSidebar button:hover {
    background-color: rgba(255,255,255,0.1);
}

/* ===== TITLE ===== */
.title {
    font-size: 36px;
    font-weight: 700;
    color: #2c3e50;
    margin-bottom: 15px;
}

/* ===== KPI CARDS ===== */
.metric-card {
    background: linear-gradient(135deg, #27ae60, #2ecc71);
    padding: 20px;
    border-radius: 14px;
    color: white;
    text-align: center;
    font-size: 18px;
    font-weight: 600;
    box-shadow: 0 6px 18px rgba(0,0,0,0.1);
    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-5px);
}

/* ===== GLASS BOX (MAIN CARDS) ===== */
.box {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    transition: 0.3s;
}

.box:hover {
    transform: scale(1.01);
}

/* ===== IMAGE STYLE ===== */
img {
    border-radius: 12px;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div > div {
    background-color: #27ae60;
}

/* ===== DOWNLOAD BUTTON ===== */
.stDownloadButton button {
    background: linear-gradient(135deg, #27ae60, #2ecc71);
    color: white;
    border-radius: 10px;
    padding: 10px 18px;
    border: none;
    font-weight: 600;
    transition: 0.3s;
}

.stDownloadButton button:hover {
    transform: scale(1.05);
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    border: 2px dashed #27ae60;
    padding: 15px;
    border-radius: 12px;
    background-color: #ffffff;
}

/* ===== CAMERA INPUT ===== */
[data-testid="stCameraInput"] {
    border: 2px dashed #3498db;
    padding: 15px;
    border-radius: 12px;
    background-color: #ffffff;
}

/* ===== SUBHEADERS ===== */
h3 {
    color: #2c3e50;
    font-weight: 600;
}

/* ===== CHART AREA ===== */
canvas {
    border-radius: 10px;
}

/* ===== FOOTER ===== */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================
# REPORT GENERATION
# =========================
def generate_report(disease, confidence):
    data = {
        "Prediction": [disease],
        "Confidence": [confidence]
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')


# =========================
# GAUGE CHART (FIXED)
# =========================
def draw_gauge(confidence):
    import matplotlib.pyplot as plt
    import numpy as np

    # safety clamp (avoid errors)
    confidence = float(confidence)
    confidence = max(0, min(confidence, 1))

    fig, ax = plt.subplots(figsize=(4, 2.5))

    # background arc
    theta = np.linspace(0, np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=10, color="#e0e0e0")

    # filled arc
    theta_fill = np.linspace(0, np.pi * confidence, 100)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), linewidth=10, color="#27ae60")

    # text
    ax.text(0, -0.2, f"{confidence*100:.2f}%", ha='center',
            fontsize=14, fontweight='bold')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.2)

    ax.axis('off')
    st.pyplot(fig)


# SIDEBAR 
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About"])


# =========================
st.markdown("""
<style>

/* =========================
   WHITE DASHBOARD BACKGROUND
========================= */
.main {
    background-color: #ffffff;
}

/* =========================
   BLUE SIDEBAR (PRO STYLE)
========================= */
section[data-testid="stSidebar"] {
    background: #1e3c72;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* =========================
   TITLE (DARK TEXT ON WHITE)
========================= */
.title {
    font-size: 28px;
    font-weight: 700;
    color: #111111;
    padding-bottom: 10px;
}

/* =========================
   INPUT CARDS (UPLOAD + CAMERA)
========================= */
div[data-testid="stFileUploader"],
div[data-testid="stCameraInput"] {
    background: #f5f7fb;
    border-radius: 14px;
    padding: 12px;
    border: 1px solid #e6e6e6;
}

/* =========================
   METRIC CARDS (LIGHT STYLE)
========================= */
.metric-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    color: #111;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    border: 1px solid #eee;
}

/* =========================
   BOX CONTAINERS
========================= */
.box {
    background: #ffffff;
    border-radius: 16px;
    padding: 18px;
    margin-top: 10px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.06);
    border: 1px solid #f0f0f0;
}

/* HEADINGS */
h1, h2, h3 {
    color: #111 !important;
}


</style>
""", unsafe_allow_html=True)

# =========================
# HOME
# =========================
if page == "Home":

    st.markdown("<div class='title'>AI Crop Disease Detection Dashboard</div>", unsafe_allow_html=True)

    st.markdown("### Input Panel")

    # =========================
    #  INPUT LAYOUT (FIXED)
    # =========================

    st.markdown("#### Upload Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    st.markdown("#### Capture Image")
    camera_image = st.camera_input("")

    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif camera_image is not None:
        image = Image.open(camera_image)

    # =========================
    # PROCESSING SECTION
    # =========================
    if image is not None:

        with st.spinner("Analyzing crop health..."):
            class_index, confidence, predictions = predict_image(image)
            disease = CLASS_NAMES[class_index]

        # =========================
        # KPI DASHBOARD ROW
        # =========================
        k1, k2, k3 = st.columns(3)

        k1.markdown(f"<div class='metric-card'>Prediction<br><b>{disease}</b></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='metric-card'>Confidence<br><b>{confidence*100:.2f}%</b></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='metric-card'>Classes<br><b>{len(CLASS_NAMES)}</b></div>", unsafe_allow_html=True)

        # =========================
        # MAIN DASHBOARD GRID
        # =========================
        left, center, right = st.columns([1.2, 1.6, 1.2])

        # LEFT PANEL
        with left:
            st.markdown("<div class='box'>", unsafe_allow_html=True)
            st.subheader("Input Image")
            st.image(image, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # CENTER PANEL
        with center:
            st.markdown("<div class='box'>", unsafe_allow_html=True)
            st.subheader("AI Prediction")

            st.markdown(f"### {disease}")

            draw_gauge(confidence)

            info = get_info(disease)

            st.subheader("Recommendation")
            st.write(info["treatment"])
            st.markdown("</div>", unsafe_allow_html=True)

        # RIGHT PANEL
        with right:
            st.markdown("<div class='box'>", unsafe_allow_html=True)
            st.subheader("Top Predictions")

            top3 = np.argsort(predictions)[-3:][::-1]

            for i in top3:
                st.write(f"{CLASS_NAMES[i]} — {predictions[i]*100:.2f}%")

            st.markdown("</div>", unsafe_allow_html=True)

        # =========================
        # INFO SECTION (FULL WIDTH)
        # =========================
        st.markdown("<div class='box'>", unsafe_allow_html=True)
        st.subheader("Disease Insight")
        st.write("Description:", info["desc"])
        st.write("Treatment:", info["treatment"])
        st.markdown("</div>", unsafe_allow_html=True)

        # =========================
        # CHART SECTION
        # =========================
        st.markdown("<div class='box'>", unsafe_allow_html=True)
        st.subheader("Prediction Probability Distribution")

        fig, ax = plt.subplots(figsize=(10,4))
        ax.barh(CLASS_NAMES, predictions)
        ax.set_xlabel("Probability")

        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        # =========================
        # DOWNLOAD
        # =========================
        csv = generate_report(disease, confidence)
        st.download_button("Download Report", csv, "report.csv")


# =========================
# ABOUT
# =========================
if page == "About":
    st.markdown("<div class='title'>About AI System</div>", unsafe_allow_html=True)
    st.write("CNN-based crop disease detection system with real-time prediction.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("<center>AI Crop Disease Detection System</center>", unsafe_allow_html=True)