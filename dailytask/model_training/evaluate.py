import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from PIL import Image
import glob
import os
from utils import predict_qr  # Import your function

# 🔹 Load ALL test images (recursively, from any subfolder)
test_images = glob.glob("dataset/test/**/*.*", recursive=True)  

# Ensure images exist
if not test_images:
    raise ValueError("❌ No test images found! Check dataset path.")

# Initialize labels (assuming folder names indicate ground truth)
y_test = []
y_pred = []

# 🔹 Predict on all test images
for img_path in test_images:
    label = 1 if "original" in img_path.lower() else 0  # Assign label based on folder name
    pred = predict_qr(img_path, threshold=0.6)  # Model prediction
    y_test.append(label)
    y_pred.append(1 if pred == "Original" else 0)

# 🔹 Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 🔹 Print results
print("🔹 Model Evaluation Results:")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-score: {f1:.4f}")
print("\n🔹 Confusion Matrix:")
print(conf_matrix)
print("\n🔹 Classification Report:")
print(report)

# 🔹 Visualizing misclassified images
misclassified_indexes = [i for i in range(len(y_pred)) if y_pred[i] != y_test[i]]

plt.figure(figsize=(10, 5))
for i, idx in enumerate(misclassified_indexes[:6]):  # Show 6 misclassified samples
    img = Image.open(test_images[idx])
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title(f"Pred: {'Original' if y_pred[idx] else 'Fake'} | Actual: {'Original' if y_test[idx] else 'Fake'}")
    plt.axis("off")

plt.tight_layout()
plt.show()
