import streamlit as st
import numpy as np
import os
os.system("uv pip install opencv-python-headless")  # Ensure the headless version is installed

import cv2  # Now import OpenCV without OpenGL dependencies



import mediapipe as mp
import tensorflow as tf
import requests
import tempfile

from gtts import gTTS
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# Load the trained model
MODEL_PATH = "sign_model_mobilenetv2.h5"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    st.error("Model file not found. Please upload the correct model.")
    st.stop()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Streamlit UI Configuration
st.set_page_config(page_title="ISL Translator", page_icon="🤟", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Go to", ["🖐️ Sign to Text/Speech", "🎤 Text/Speech to Sign"])

st.title("🤟 Indian Sign Language Translator")
st.write("A real-time ISL translation system supporting both directions.")

# Helper functions
def preprocess_hand(image):
    image_resized = cv2.resize(image, (128, 128)) / 255.0
    return np.expand_dims(image_resized, axis=0)

def predict_sign(image):
    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    return predicted_index

def get_sign_label(index):
    try:
        response = requests.get(f"https://www.signasl.org/sign/{index}")
        if response.status_code == 200:
            return response.json().get("label", "Unknown Sign")
    except:
        pass
    return "Unknown Sign"

def fetch_sign_image(word):
    try:
        response = requests.get(f"https://www.signasl.org/sign/{word}")
        if response.status_code == 200:
            data = response.json()
            return data.get("image_url", None)
    except:
        pass
    return None

# ========== SIGN TO TEXT/SPEECH ==========
if selected_tab == "🖐️ Sign to Text/Speech":
    st.subheader("📸 Upload an Image or Use Webcam")
    uploaded_file = st.file_uploader("Upload a sign language image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Uploaded Image")

        predicted_index = predict_sign(preprocess_hand(image))
        predicted_sign = get_sign_label(predicted_index)
        st.success(f"**Predicted Sign:** {predicted_sign}")
        
        # Convert Prediction to Speech
        tts = gTTS(text=predicted_sign, lang="en")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        st.audio(temp_audio.name, format="audio/mp3")

# ========== TEXT/SPEECH TO SIGN ==========
if selected_tab == "🎤 Text/Speech to Sign":
    st.subheader("📝 Enter Text or Speak to Convert into Sign Language")
    text_input = st.text_input("Enter Text")
    
    if text_input:
        st.write(f"🔠 Converting **'{text_input}'** to sign language...")
        words = text_input.lower().split()
        for word in words:
            image_url = fetch_sign_image(word)
            if image_url:
                st.image(image_url, caption=word.capitalize())
            else:
                st.warning(f"No sign found for: {word}")
        
        # Convert Text to Speech
        tts = gTTS(text=text_input, lang="en")
        temp_audio_text = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio_text.name)
        st.audio(temp_audio_text.name, format="audio/mp3")
