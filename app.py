import os
import streamlit as st
import numpy as np
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
from PIL import Image
import cv2

# Install missing dependencies (for cloud-based deployments)
os.system("apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0")

# Load trained model
model = tf.keras.models.load_model("sign_model_mobilenetv2.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define label mapping (Update as per dataset)
label_map = {0: "Hello", 1: "Thank You", 2: "Please"}  # Ensure it matches your dataset

# Function to preprocess the image before prediction
def preprocess_hand(image):
    image_resized = cv2.resize(image, (128, 128)) / 255.0
    return np.expand_dims(image_resized, axis=0)

# Streamlit UI
st.title("🤟 Indian Sign Language Recognition")
st.sidebar.header("📷 Camera Options")

# Capture image using Streamlit's built-in camera input
img_file = st.camera_input("Take a picture")

if img_file is not None:
    # Convert image to OpenCV format
    image = Image.open(img_file)
    image = np.array(image)
    
    # Convert from RGB to BGR (OpenCV format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Process with MediaPipe
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box around the hand
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * image.shape[1]
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * image.shape[0]
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * image.shape[1]
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * image.shape[0]

            hand_img = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                hand_img = preprocess_hand(hand_img)

                # Predict the sign
                prediction = model.predict(hand_img)
                predicted_index = np.argmax(prediction)
                predicted_sign = label_map.get(predicted_index, "Unknown")

                # Display prediction
                st.sidebar.subheader(f"🖐 Predicted Sign: **{predicted_sign}**")

                # Convert text to speech
                tts = gTTS(text=predicted_sign, lang="en")
                tts.save("output.mp3")
                st.audio("output.mp3", autoplay=True)

    # Display processed image
    st.image(image, channels="BGR", caption="Processed Image")
