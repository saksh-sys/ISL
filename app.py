import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
import os

# GitHub link to your model
MODEL_URL = "https://github.com/saksh-sys/ISL/blob/main/model_sign_99.h5?raw=true"

# Download model if not exists
MODEL_PATH = "model_sign_99.h5"
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (update based on your dataset)
class_labels = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]  # A-Z + 0-9

# Streamlit UI
st.title("Indian Sign Language Recognition")
st.write("Upload an image of a hand gesture, and the model will predict the corresponding letter or number.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Display results
    st.image(img, caption=f"Predicted Sign: {predicted_class}", use_column_width=True)
    st.success(f"Predicted Sign: {predicted_class}")
