import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import speech_recognition as sr
from gtts import gTTS
from PIL import Image
import tempfile
import os

# Load the trained model
model = tf.keras.models.load_model("sign_model_mobilenetv2.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Class Labels (Modify based on dataset)
label_map = {0: "Hello", 1: "Thank You", 2: "Yes", 3: "No", 4: "I Love You"}

# Streamlit UI Configuration
st.set_page_config(page_title="ISL Translator", page_icon="🤟", layout="wide")

st.title("🤟 Indian Sign Language Translator")
st.write("Translate **sign language** to text/speech and vice versa!")

# Navigation Tabs
tab1, tab2 = st.tabs(["🖐️ Sign to Text/Speech", "🎤 Text/Speech to Sign"])

# ======= SIGN TO TEXT/SPEECH =======
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
            return label_map.get(predicted_index, "Unknown Sign")

        predicted_sign = predict_sign(preprocess_hand(image))
        st.success(f"**Predicted Sign:** {predicted_sign}")

        # Convert Prediction to Speech
        tts = gTTS(text=predicted_sign, lang="en")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        st.audio(temp_audio.name, format="audio/mp3")

# ======= TEXT/SPEECH TO SIGN =======
with tab2:
    st.subheader("📝 Enter Text or Speak to Convert into Sign Language")

    # Text Input
    text_input = st.text_input("Enter Text")

    # Speech Input
    st.write("🎤 Click below to record your voice")
    if st.button("Start Recording"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            try:
                audio = recognizer.listen(source, timeout=5)
                text_input = recognizer.recognize_google(audio)
                st.success(f"Recognized Speech: {text_input}")
            except sr.UnknownValueError:
                st.error("Could not understand the audio.")
            except sr.RequestError:
                st.error("Error connecting to speech recognition service.")

    # Convert Text to Sign Language
    if text_input:
        st.write(f"🔠 Converting **'{text_input}'** to sign language...")

        # Load Sign Language Images (Ensure the correct folder structure)
        sign_images = {
            "hello": "sign_images/hello.jpg",
            "thank you": "sign_images/thank_you.jpg",
            "yes": "sign_images/yes.jpg",
            "no": "sign_images/no.jpg",
            "i love you": "sign_images/i_love_you.jpg",
        }

        # Display corresponding sign images
        word_list = text_input.lower().split()
        for word in word_list:
            if word in sign_images and os.path.exists(sign_images[word]):  # Check if file exists
                st.image(sign_images[word], caption=word.capitalize(), use_container_width=False)
            else:
                st.warning(f"No sign found for: {word}")

        # Convert Text to Speech
        tts = gTTS(text=text_input, lang="en")
        temp_audio_text = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio_text.name)
        st.audio(temp_audio_text.name, format="audio/mp3")
