import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
import os

# Load trained model
MODEL_PATH = "sign_model_mobilenetv2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define label mapping (Ensure this matches your dataset)
label_map = {i: chr(65 + i) for i in range(26)}  # A-Z
label_map.update({26 + i: str(i + 1) for i in range(9)})  # 1-9

# Function to preprocess hand image
def preprocess_hand(image):
    image_resized = cv2.resize(image, (128, 128)) / 255.0
    return np.expand_dims(image_resized, axis=0)

def main():
    st.title("🤟 Real-Time Sign to Text Conversion")

    # Store detected text across frames
    if "detected_text" not in st.session_state:
        st.session_state.detected_text = ""

    # Display detected text (READ-ONLY)
    st.write("### 📄 Recognized Text:")
    st.text_area("Detected Letters:", st.session_state.detected_text, height=100, disabled=True)

    # Streamlit container for live video feed
    stframe = st.empty()

    # Open camera automatically
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("⚠ Could not access camera. Please check camera permissions.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠ Failed to capture image.")
            break

        # Convert frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract bounding box for the hand
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]

                # Crop and preprocess hand region
                hand_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                    hand_img = preprocess_hand(hand_img)

                    # Predict the sign
                    prediction = model.predict(hand_img)
                    predicted_index = np.argmax(prediction)
                    predicted_sign = label_map.get(predicted_index, "")

                    # Append detected letter/number
                    if predicted_sign:
                        st.session_state.detected_text += predicted_sign
                        st.experimental_rerun()  # Rerun the UI to update text live

        # Show real-time camera feed
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()

    # Convert text to speech
    if st.button("🔊 Convert to Speech"):
        if st.session_state.detected_text:
            tts = gTTS(text=st.session_state.detected_text, lang="en")
            speech_path = "speech_output.mp3"
            tts.save(speech_path)
            st.audio(speech_path, format="audio/mp3", autoplay=True)
        else:
            st.warning("⚠ No text detected yet.")

    # Clear detected text
    if st.button("🗑 Clear Text"):
        st.session_state.detected_text = ""
        st.experimental_rerun()  # Refresh UI to clear text

if __name__ == "__main__":
    main()
