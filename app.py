import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
from sklearn.svm import SVC  

st.set_page_config(page_title="Chest X-Ray Pneumonia Detection", layout="centered")

# ---- Load models ----
LR_PATH = 'models/logistic_regression_model.pkl'
CNN_PATH = 'models/cnn_model.h5'
SVM_PATH = 'models1/svm_model.pkl'

# load logistic regression
try:
    with open(LR_PATH, 'rb') as f:
        lr_model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load Logistic Regression model: {e}")
    lr_model = None

# load cnn
try:
    cnn_model = load_model(CNN_PATH)
except Exception as e:
    st.error(f"Failed to load CNN model: {e}")
    cnn_model = None

# load svm
try:
    with open(SVM_PATH, 'rb') as f:
        svm_model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load SVM model: {e}")
    svm_model = None

# ---- UI ----
st.title("Chest X-Ray Pneumonia Detection")
st.markdown(
    "**Upload one or more X-Ray images**, and this app will predict whether each image shows **Pneumonia** or **Normal**. "
    "This app shows predictions from CNN, Logistic Regression, and SVM (if loaded)."
)

def preprocess_image(image: Image.Image):
    """
    Input: PIL Image
    Returns: (img_cnn, img_flat) or (None, None) on error
    img_cnn -> shape (1,150,150,1) float32 normalized
    img_flat -> shape (1,22500) float32 normalized
    """
    try:
        if image.mode != "L":
            image = image.convert("L")
        image = image.resize((150, 150))
        img_array = np.array(image).astype(np.float32) / 255.0
        img_cnn = img_array.reshape(1, 150, 150, 1)
        img_flat = img_array.flatten().reshape(1, -1)
        return img_cnn, img_flat
    except Exception as e:
        st.error(f"Error processing an image: {e}")
        return None, None

uploaded_files = st.file_uploader("Upload one or more X-ray Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    images = []
    for file in uploaded_files:
        try:
            img = Image.open(file)
            images.append(img)
        except Exception as e:
            st.error(f"Error loading file {file.name}: {e}")

    st.subheader("Preview Images")
    cols = st.columns(min(len(images), 4))
    for i, img in enumerate(images):
        with cols[i % len(cols)]:
            st.image(img, caption=f"Image {i + 1}", use_column_width=True)

    if st.button("Classify Images"):
        results = []
        for i, img in enumerate(images):
            st.write(f"Classifying Image {i + 1}...")
            img_cnn, img_flat = preprocess_image(img)

            if img_cnn is None or img_flat is None:
                results.append({
                    "Image": f"Image {i + 1}",
                    "CNN": "Error",
                    "Logistic Regression": "Error",
                    "SVM": "Error"
                })
                continue

            # CNN prediction -> probability for label 1 (NORMAL) because training used labels: 0=PNEUMONIA,1=NORMAL
            if cnn_model is not None:
                try:
                    prob = float(cnn_model.predict(img_cnn)[0][0])
                    cnn_label = 1 if prob > 0.5 else 0
                    cnn_text = "Normal" if cnn_label == 1 else "Pneumonia"
                except Exception as e:
                    cnn_text = f"Error: {e}"
            else:
                cnn_text = "Model not loaded"

            # Logistic Regression prediction (expects same label encoding: 0=PNEUMONIA,1=NORMAL)
            if lr_model is not None:
                try:
                    lr_pred = lr_model.predict(img_flat)
                    lr_label = int(lr_pred[0]) if hasattr(lr_pred, '__len__') else int(lr_pred)
                    lr_text = "Normal" if lr_label == 1 else "Pneumonia"
                except Exception as e:
                    lr_text = f"Error: {e}"
            else:
                lr_text = "Model not loaded"

            # SVM prediction
            if svm_model is not None:
                try:
                    svm_pred = svm_model.predict(img_flat)
                    svm_label = int(svm_pred[0]) if hasattr(svm_pred, '__len__') else int(svm_pred)
                    svm_text = "Normal" if svm_label == 1 else "Pneumonia"
                except Exception as e:
                    svm_text = f"Error: {e}"
            else:
                svm_text = "Model not loaded"

            results.append({
                "Image": f"Image {i + 1}",
                "CNN": cnn_text,
                "Logistic Regression": lr_text,
                "SVM": svm_text
            })

        st.subheader("Prediction Results")
        for res in results:
            st.write(f"**{res['Image']}**")
            st.write(f"- CNN Prediction: {res['CNN']}")
            st.write(f"- Logistic Regression Prediction: {res['Logistic Regression']}")
            st.write(f"- SVM Prediction: {res['SVM']}")
else:
    st.info("Please upload one or more images to classify.")
