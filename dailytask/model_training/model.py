import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths to your dataset folders
train_dir = "dataset/train"
test_dir = "dataset/test"

# Image size & batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Data augmentation & preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

# Load ResNet50 (without top layers)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False

# Custom layers
x = Flatten()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dense(1, activation="sigmoid")(x)  # Binary classification

# Create model
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
EPOCHS = 10
model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS)

# Save the trained model
model.save("model_training/QR_Authentication_ResNet50.keras")

print("Model training complete and saved successfully!")
