import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("model_sign_99.h5")

# Extract class labels dynamically from the dataset folder
DATASET_PATH = "dataset"
class_labels = sorted(os.listdir(DATASET_PATH))  # Ensure sorted order

def predict_sign(img_file):
    """Predicts the sign from an uploaded or captured image."""
    img = Image.open(img_file).convert("RGB")  # Ensure RGB format
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]  # Map prediction index to class

    return predicted_class
