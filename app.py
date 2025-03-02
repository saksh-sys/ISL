import os

# Install system dependencies (for cloud environments)
os.system("apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0")

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

# Load trained model
model = tf.keras.models.load_model("sign_model_mobilenetv2.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define label mapping
label_map = {0: "Hello", 1: "Thank You", 2: "Please"}  # Update with actual labels

# Preprocess function
def preprocess_hand(image):
    image_resized = cv2.resize(image, (128, 128)) / 255.0
    return np.expand_dims(image_resized, axis=0)

# Streamlit UI
st.title("Indian Sign Language Recognition")
st.sidebar.header("Camera Options")

# Open webcam using Streamlit
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]

            hand_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                hand_img = preprocess_hand(hand_img)

                prediction = model.predict(hand_img)
                predicted_index = np.argmax(prediction)
                predicted_sign = label_map.get(predicted_index, "Unknown")

                st.sidebar.subheader(f"Predicted Sign: {predicted_sign}")

                # Convert text to speech
                tts = gTTS(text=predicted_sign, lang="en")
                tts.save("output.mp3")
                audio = AudioSegment.from_mp3("output.mp3")
                play(audio)

    frame_placeholder.image(frame, channels="BGR")

cap.release()
st.write("Press 'Stop' to exit")
