import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import tempfile
import os
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the trained model
MODEL_PATH = "sign_model_mobilenetv2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define label mapping (Ensure this matches dataset)
label_map = {i: chr(65 + i) for i in range(26)}  # A-Z
label_map.update({26 + i: str(i + 1) for i in range(9)})  # 1-9

# Initialize session state for storing detected text
if "detected_text" not in st.session_state:
    st.session_state.detected_text = ""

# Function to preprocess the hand image
def preprocess_hand(image):
    image_resized = cv2.resize(image, (128, 128)) / 255.0
    return np.expand_dims(image_resized, axis=0)

# Streamlit Video Transformer for Real-time Detection
class SignLanguageTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect hands
        result = hands.process(rgb_image)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get bounding box of hand
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * image.shape[1]
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * image.shape[0]
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * image.shape[1]
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * image.shape[0]

                # Crop and preprocess the hand region
                hand_img = image[int(y_min):int(y_max), int(x_min):int(x_max)]
                if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                    hand_img = preprocess_hand(hand_img)

                    # Predict the sign
                    prediction = model.predict(hand_img)
                    predicted_index = np.argmax(prediction)
                    predicted_sign = label_map.get(predicted_index, "")

                    # Append the detected letter to session state text
                    if predicted_sign:
                        st.session_state.detected_text += predicted_sign

        return image

# Streamlit UI
def main():
    st.title("🤟 Sign to Text Conversion (Live Detection)")

    # Start Webcam
    st.write("### 📷 Live Camera Feed")
    webrtc_streamer(key="sign_detect", video_transformer_factory=SignLanguageTransformer)

    # Read-only text box to display detected text
    st.write("### ✍ Recognized Text")
    st.text_area("Detected Text:", value=st.session_state.detected_text, height=100, disabled=True)

    # Buttons
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🔄 Clear Text"):
            st.session_state.detected_text = ""

    with col2:
        if st.button("🔊 Convert to Speech"):
            if st.session_state.detected_text:
                tts = gTTS(text=st.session_state.detected_text, lang="en")
                tts.save("output.mp3")
                st.audio("output.mp3", format="audio/mp3", autoplay=True)

if __name__ == "__main__":
    main()
