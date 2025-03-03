import os
import streamlit as st
import cv2
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO

# Define dataset path
DATASET_PATH = "dataset"

# Function to convert speech-to-text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio).upper()
            st.success(f"Recognized Text: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
        except sr.RequestError:
            st.error("STT request failed. Check internet connection.")
    return ""

# Function to generate speech from text
def text_to_speech(text):
    tts = gTTS(text)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    st.audio(audio_buffer, format="audio/mp3")

# Function to display ISL images
def text_to_sign_images(text):
    images = []
    for char in text:
        img_path = os.path.join(DATASET_PATH, char + ".jpg")  # Assuming .jpg images
        if os.path.exists(img_path):
            images.append(img_path)
    
    cols = st.columns(len(images) if images else 1)
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cols[i].image(img, use_column_width=True)

# Streamlit UI
def main():
    st.title("📖 Text/Speech to Sign Language")
    
    option = st.radio("Choose Input Method:", ["📝 Type Text", "🎤 Speak"])
    text_input = ""
    
    if option == "📝 Type Text":
        text_input = st.text_input("Enter text:")
    elif option == "🎤 Speak":
        if st.button("🎙 Start Listening"):
            text_input = speech_to_text()
    
    if text_input:
        st.subheader("🖼️ Sign Language Representation")
        text_to_sign_images(text_input)
        
        st.subheader("🔊 Speech Output")
        text_to_speech(text_input)

if __name__ == "__main__":
    main()
