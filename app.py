import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import requests
import tempfile
import os
from gtts import gTTS
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# Load the trained model
model = tf.keras.models.load_model("sign_model_mobilenetv2.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Streamlit UI Configuration
st.set_page_config(page_title="ISL Translator", page_icon="🤟", layout="wide")

st.title("🤟 Indian Sign Language Translator")
st.write("Translate **sign language** to text/speech and vice versa!")

# Tabs for different modes
tab1, tab2 = st.tabs(["🖐️ Sign to Text/Speech", "🎤 Text/Speech to Sign"])

# ========== SIGN TO TEXT/SPEECH ==========
with tab1:
    st.subheader("📸 Upload an Image or Use Webcam")

    # Upload Image
    uploaded_file = st.file_uploader("Upload a sign language image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Convert to OpenCV format
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the Image
        def preprocess_hand(image):
            image_resized = cv2.resize(image, (128, 128)) / 255.0
            return np.expand_dims(image_resized, axis=0)

        # Predict the Sign
        def predict_sign(image):
            prediction = model.predict(image)
            predicted_index = np.argmax(prediction)
            return predicted_index

        predicted_index = predict_sign(preprocess_hand(image))
        
        # Fetch label from web API instead of fixed mapping
        def get_sign_label(index):
            response = requests.get(f"https://api.signlanguage.com/labels/{index}")
            if response.status_code == 200:
                return response.json().get("label", "Unknown Sign")
            return "Unknown Sign"

        predicted_sign = get_sign_label(predicted_index)
        st.success(f"**Predicted Sign:** {predicted_sign}")

        # Convert Prediction to Speech
        tts = gTTS(text=predicted_sign, lang="en")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        st.audio(temp_audio.name, format="audio/mp3")

# ========== TEXT/SPEECH TO SIGN ==========
with tab2:
    st.subheader("📝 Enter Text or Speak to Convert into Sign Language")

    # Text Input
    text_input = st.text_input("Enter Text")

    # Speech Input (Using WebRTC)
    def audio_callback(frame):
        return frame

    st.write("🎤 Click below to record your voice")
    webrtc_streamer(key="speech", mode=WebRtcMode.SENDRECV, audio_processor_factory=audio_callback)

    # Convert Text to Sign Language
    if text_input:
        st.write(f"🔠 Converting **'{text_input}'** to sign language...")

        # Fetch sign language images dynamically
        def fetch_sign_image(word):
            response = requests.get(f"https://api.signlanguage.com/signs/{word}")
            if response.status_code == 200:
                return response.json().get("image_url", None)
            return None

        words = text_input.lower().split()
        for word in words:
            image_url = fetch_sign_image(word)
            if image_url:
                st.image(image_url, caption=word.capitalize(), use_container_width=False)
            else:
                st.warning(f"No sign found for: {word}")

        # Convert Text to Speech
        tts = gTTS(text=text_input, lang="en")
        temp_audio_text = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio_text.name)
        st.audio(temp_audio_text.name, format="audio/mp3")
