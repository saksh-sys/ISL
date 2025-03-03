import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
from io import BytesIO
import os
from sign_to_text import predict_sign

# Load the model from GitHub
MODEL_URL = "https://github.com/saksh-sys/ISL/blob/main/model_sign_99.h5?raw=true"
MODEL_PATH = "model_sign_99.h5"

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Streamlit UI
st.title("Indian Sign Language Recognition")
st.write("Upload an image to classify the sign language letter or number.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption="Uploaded Image", use_column_width=True)
    
    # Process image
    img = image_data.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Prediction
    predictions = model.predict(img_array)
    predicted_class = predict_sign(predictions)
    
    # Show results
    st.subheader(f"Predicted Sign: {predicted_class}")
