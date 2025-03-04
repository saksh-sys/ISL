import cv2
import streamlit as st
import numpy as np
from sign_to_text import predict_sign
from PIL import Image

st.title("Indian Sign Language Recognition")
st.write("Real-time sign language detection using your webcam.")

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
stframe = st.empty()

# Initialize session state
if "captured_text" not in st.session_state:
    st.session_state.captured_text = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image")
        break

    # Convert frame to PIL format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Predict sign
    predicted_class = predict_sign(pil_image)

    # Update text
    if predicted_class:
        st.session_state.captured_text += predicted_class

    # Display frame
    stframe.image(frame_rgb, caption=f"Predicted Sign: {predicted_class}", use_column_width=True)

# Release camera
cap.release()
