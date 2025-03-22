import streamlit as st
import numpy as np
import tensorflow as tf
import gdown
import os
import zipfile
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Google Drive File IDs
MODEL_FILE_ID = "1p_0qGwXEOeA690jyvwllworwj75WdFdt"  # Update with correct model file ID
DATASET_FILE_ID = "1YF-GiDnT0jvDRvICM8xETegtmmnFJ3dw"  # Correct dataset file ID

MODEL_PATH = "QR_Authentication_ResNet50.keras"
DATASET_ZIP_PATH = "dataset.zip"
DATASET_EXTRACT_PATH = "dataset"

# Download Model if not exists
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Download and Extract Dataset if not exists
if not os.path.exists(DATASET_EXTRACT_PATH):
    st.info("Downloading dataset...")
    gdown.download(f"https://drive.google.com/uc?id={DATASET_FILE_ID}", DATASET_ZIP_PATH, quiet=False)

    st.info("Extracting dataset...")
    with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")
    
    os.remove(DATASET_ZIP_PATH)  # Clean up

# Function to Evaluate Model
def evaluate_model(test_dir, label):
    true_labels = []
    predictions = []

    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        predicted_label = 1 if prediction > 0.5 else 0

        true_labels.append(label)
        predictions.append(predicted_label)

    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions)
    rec = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return acc, prec, rec, f1

# Streamlit UI
st.title("QR Code Authentication with ResNet50")

uploaded_file = st.file_uploader("Upload a QR Code Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    result = "Original" if prediction > 0.5 else "Counterfeit"

    st.subheader(f"Prediction: **{result}**")

# Evaluate Model Button
if st.button("Evaluate Model Accuracy"):
    original_test_dir = os.path.join(DATASET_EXTRACT_PATH, "test/original")
    counterfeit_test_dir = os.path.join(DATASET_EXTRACT_PATH, "test/counterfeit")

    if not os.path.exists(original_test_dir) or not os.path.exists(counterfeit_test_dir):
        st.error("Test dataset not found!")
    else:
        acc1, prec1, rec1, f11 = evaluate_model(original_test_dir, 1)
        acc2, prec2, rec2, f12 = evaluate_model(counterfeit_test_dir, 0)

        avg_acc = (acc1 + acc2) / 2
        avg_prec = (prec1 + prec2) / 2
        avg_rec = (rec1 + rec2) / 2
        avg_f1 = (f11 + f12) / 2

        st.subheader("Model Evaluation Results:")
        st.write(f"**Accuracy:** {avg_acc:.4f}")
        st.write(f"**Precision:** {avg_prec:.4f}")
        st.write(f"**Recall:** {avg_rec:.4f}")
        st.write(f"**F1 Score:** {avg_f1:.4f}")
