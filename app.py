import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
import os
import cv2
from PIL import Image
from sign_to_text import predict_sign
from text_to_sign import text_to_sign

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
page = st.sidebar.radio("Go to", ["Sign to Text", "Text to Sign", "About"])

if page == "Sign to Text":
    st.title("Indian Sign Language Recognition")
    st.write("Upload an image or capture from webcam to predict the corresponding sign.")

    # Container for prediction
    container = st.container()
    captured_text = st.text_area("Predicted Text:", value="", height=100, disabled=True)

    # Webcam capture
    cam = st.camera_input("Capture an image")

    # Process webcam image
    if cam is not None:
        predicted_class = predict_sign(cam)
        st.image(cam, caption="Captured Image", use_container_width=True)
        st.success(f"Predicted Sign: {predicted_class}")

        # Append predicted sign to the text box
        captured_text += predicted_class
        st.text_area("Predicted Text:", value=captured_text, height=100, disabled=True)

    # Upload image
    uploaded_file = st.file_uploader("Or upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        predicted_class = predict_sign(uploaded_file)
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        st.success(f"Predicted Sign: {predicted_class}")

        # Append predicted sign to the text box
        captured_text += predicted_class
        st.text_area("Predicted Text:", value=captured_text, height=100, disabled=True)

    # Clear text button
    if st.button("Clear Text"):
        captured_text = ""
        st.text_area("Predicted Text:", value=captured_text, height=100, disabled=True)

elif page == "Text to Sign":
    st.title("Text to Sign Language")
    st.write("Enter text, and the corresponding sign language images will be displayed.")

    user_text = st.text_input("Enter text:", "")
    
    if st.button("Convert"):
        if user_text:
            images = text_to_sign(user_text)
            for img in images:
                st.image(img, use_column_width=True)

elif page == "About":
    st.title("About This Project")
    st.write(
        """
        This application recognizes **Indian Sign Language (ISL)** gestures using deep learning.  
        - **Dataset**: Indian Sign Language (ISLRTC)  
        - **Model**: MobileNetV2  
        - **Framework**: TensorFlow & Streamlit  

        **How It Works:**
        - Capture or upload an image of a sign.
        - The model predicts the corresponding letter/number.
        - Text-to-sign conversion available.

        **Built by:** [Saksham Raj Gupta](https://github.com/saksh-sys)
        """
    )
