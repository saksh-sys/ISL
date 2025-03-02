import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
import tempfile
import os
import time

# Load trained model
MODEL_PATH = "sign_model_mobilenetv2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Label Mapping (Ensure it matches your dataset)
label_map = {i: chr(65 + i) for i in range(26)}  # A-Z
label_map.update({26 + i: str(i + 1) for i in range(9)})  # 1-9

# Streamlit UI
st.title("🤟 Indian Sign Language Recognition")
st.write("### 📷 Camera Feed & Real-time Sign Detection")

# Initialize session state for detected text
if "detected_text" not in st.session_state:
    st.session_state.detected_text = ""

# Create a text box (readonly)
st.text_area("📄 Recognized Text:", value=st.session_state.detected_text, height=100, key="textbox", disabled=True)

# Button to clear text box
if st.button("🗑 Clear Text"):
    st.session_state.detected_text = ""

# Open webcam using OpenCV
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Could not access the camera. Make sure it's connected.")
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]

            # Crop and preprocess the hand image
            hand_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                hand_img = cv2.resize(hand_img, (128, 128)) / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                # Predict the sign
                prediction = model.predict(hand_img)
                predicted_index = np.argmax(prediction)
                predicted_sign = label_map.get(predicted_index, "")

                # Append detected sign to textbox
                if predicted_sign:
                    st.session_state.detected_text += predicted_sign

                # Display predicted sign
                cv2.putText(frame, f"Detected: {predicted_sign}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display webcam feed
    frame_placeholder.image(frame, channels="BGR", use_column_width=True)

    # Stop when user exits
    if st.button("❌ Stop Camera"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
