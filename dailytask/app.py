import os
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import model_training.model as model_module  # Import your model from the model_training directory

# Paths
dataset_path = "dataset/test"
model_path = "model_training/model.pth"

# Load model
def load_model():
    model = model_module.YourModelClass()  # Replace with your model class
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Image transformation
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Model Evaluation
def evaluate_model():
    if not os.path.exists(dataset_path):
        st.error("Dataset folder not found! Make sure 'dataset/test' exists.")
        return
    
    images = os.listdir(dataset_path)
    if not images:
        st.warning("No images found in dataset/test.")
        return
    
    model = load_model()
    results = {}
    for img_name in images:
        img_path = os.path.join(dataset_path, img_name)
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform_image(image)
        
        with torch.no_grad():
            output = model(img_tensor)
            predicted_class = torch.argmax(output, 1).item()
        
        results[img_name] = predicted_class
    
    return results

# Streamlit UI
st.title("AI Model Evaluation")
if st.button("Evaluate Model"):
    results = evaluate_model()
    if results:
        st.write("### Evaluation Results:")
        for img, label in results.items():
            st.write(f"{img}: Predicted Class {label}")
