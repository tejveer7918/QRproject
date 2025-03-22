import streamlit as st
import gdown
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import shutil

# Google Drive file ID of the model
MODEL_ID = "1p_0qGwXEOeA690jyvwllworwj75WdFdt"
MODEL_PATH = "QR_Authentication_ResNet50.keras"

# Download model if not already downloaded
if not os.path.exists(MODEL_PATH):
    st.write("📥 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)

# Load the model
st.write("🔄 Loading model...")
model = load_model(MODEL_PATH)
st.success("✅ Model Loaded Successfully!")

# Class Labels (Adjust according to dataset)
CLASS_NAMES = {0: "Fake QR Code", 1: "Original QR Code"}

# Function to predict a single image
def predict_qr(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust input size if needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    prediction = model.predict(img_array)
    predicted_label = int(np.round(prediction[0][0]))  # Adjust threshold if needed
    return predicted_label

# Function to evaluate model accuracy
def evaluate_model(test_dir, true_label):
    y_true = []
    y_pred = []

    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        predicted_label = predict_qr(img_path)
        
        y_true.append(true_label)  # 1 for Original, 0 for Fake
        y_pred.append(predicted_label)

    # Compute Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1

# Streamlit UI
st.title("🔍 QR Code Authentication using ResNet50")

# Upload image for prediction
uploaded_file = st.file_uploader("📤 Upload a QR Code Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Save uploaded file
    img_path = "uploaded_qr.png"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict and display result
    predicted_label = predict_qr(img_path)
    st.image(img_path, caption=f"Prediction: {CLASS_NAMES[predicted_label]}", use_column_width=True)

# Evaluate model accuracy
if st.button("Evaluate Model"):
    st.write("🔍 Evaluating model on test dataset...")

    original_test_dir = "dataset/test/original"  # Update with actual path
    fake_test_dir = "dataset/test/fake"          # Update with actual path

    # Evaluate Original QRs (Label = 1)
    st.write("📌 Evaluating on **Original QR Codes**")
    acc1, prec1, rec1, f11 = evaluate_model(original_test_dir, 1)

    # Evaluate Fake QRs (Label = 0)
    st.write("📌 Evaluating on **Fake QR Codes**")
    acc2, prec2, rec2, f12 = evaluate_model(fake_test_dir, 0)

    # Compute Overall Metrics
    overall_acc = (acc1 + acc2) / 2
    overall_prec = (prec1 + prec2) / 2
    overall_rec = (rec1 + rec2) / 2
    overall_f1 = (f11 + f12) / 2

    st.write(f"🎯 **Overall Model Accuracy: {overall_acc:.4f}**")
    st.write(f"✅ Precision: {overall_prec:.4f}")
    st.write(f"🔁 Recall: {overall_rec:.4f}")
    st.write(f"🏆 F1 Score: {overall_f1:.4f}")

st.success("🚀 App Ready! Upload a QR code or evaluate model accuracy.")
