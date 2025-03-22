import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("model_training/QR_Authentication_ResNet50.keras")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_qr(img_path, threshold=0.6):
    img = preprocess_image(img_path)
    prediction = model.predict(img)[0][0]
    return "Original" if prediction > threshold else "Fake"
