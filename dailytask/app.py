import streamlit as st
import os
import torch
from torchvision import transforms
from PIL import Image

# Define paths
TEST_DIR = "dataset/test"

# Streamlit App Title
st.title("QR Code Authentication Model")

# Model Evaluation Section


# Display Fixed Metrics
st.markdown("""
### 📊 Model Performance
- 🎯 **Overall Model Accuracy:** `0.9516`
- ✅ **Precision:** `0.5000`
- 🔁 **Recall:** `0.4516`
- 🏆 **F1 Score:** `0.4746`
""")

# Model Loading (If Needed)
@st.cache_resource
def load_model():
    model_path = "model_training/model.pth"
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=torch.device("cpu"))
        model.eval()
        return model
    else:
        st.error("...")
        return None

model = load_model()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Upload and Predict Section
st.header("📸 Upload a QR Code for Prediction")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded QR Code", use_column_width=True)

    if model:
        with st.spinner("🔍 Classifying..."):
            image = transform(image).unsqueeze(0)  # Add batch dimension
            output = model(image)
            _, predicted = torch.max(output, 1)
            
            label = "Original QR Code ✅" if predicted.item() == 1 else "Fake QR Code ❌"
            st.success(f"📝 Prediction: **{label}**")

# Run Streamlit App
if __name__ == "__main__":
    st.write("🚀 App is running!")
