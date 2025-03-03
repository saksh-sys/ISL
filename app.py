import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
import os
from sign_to_text import predict_sign

# GitHub link to your model
MODEL_URL = "https://github.com/saksh-sys/ISL/blob/main/model_sign_99.h5?raw=true"

# Download model if not exists
MODEL_PATH = "model_sign_99.h5"
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About"])

if page == "Home":
    # Main Page - Sign Prediction
    st.title("Indian Sign Language Recognition")
    st.write("Upload an image of a hand gesture, and the model will predict the corresponding letter or number.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Make prediction
        predicted_class = predict_sign(uploaded_file)

        # Display results
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        st.success(f"Predicted Sign: {predicted_class}")

elif page == "About":
    # About Page
    st.title("About This Project")
    st.write(
        """
        This application is built to recognize **Indian Sign Language (ISL)** gestures using a deep learning model. 
        The model is trained on a dataset containing **36 classes (A-Z & 0-9)** using **MobileNetV2**.

        🔹 **Dataset**: Indian Sign Language (ISLRTC)  
        🔹 **Model**: Pre-trained MobileNetV2 + Custom Classification  
        🔹 **Framework**: TensorFlow & Streamlit  

        ### How It Works:
        - Upload an image of a hand gesture.
        - The model processes and predicts the corresponding sign.
        - Results are displayed instantly with high accuracy!

        **Built by:** [Saksham Raj Gupta](https://github.com/saksh-sys)
        """
    )
