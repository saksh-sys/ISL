import os
import streamlit as st
import cv2
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from io import BytesIO

# Define dataset path (update if needed)
DATASET_PATH = "dataset"

# Function to convert uploaded audio file to WAV (required for STT)
def convert_audio_to_wav(uploaded_file):
    audio = AudioSegment.from_file(uploaded_file)
    wav_buffer = BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    return wav_buffer

# Function to convert speech-to-text from uploaded file
def speech_to_text_from_file(audio_buffer):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_buffer) as source:
        st.info("Processing audio file...")
        audio = recognizer.record(source)
        try:
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
        char_folder = os.path.join(DATASET_PATH, char)
        if os.path.exists(char_folder):
            image_files = os.listdir(char_folder)
            if image_files:
                img_path = os.path.join(char_folder, image_files[0])  # Use first image
                images.append(img_path)
    
    # Display images in Streamlit
    cols = st.columns(len(images) if images else 1)
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cols[i].image(img, use_column_width=True)

# Streamlit UI
def main():
    st.title("📖 Text/Speech to Sign Language")
    
    option = st.radio("Choose Input Method:", ["📝 Type Text", "📤 Upload Audio File"])
    text_input = ""
    
    if option == "📝 Type Text":
        text_input = st.text_input("Enter text:")
    elif option == "📤 Upload Audio File":
        uploaded_file = st.file_uploader("Upload an audio file (MP3 or AAC)", type=["mp3", "aac"])
        if uploaded_file is not None:
            wav_audio = convert_audio_to_wav(uploaded_file)
            text_input = speech_to_text_from_file(wav_audio)
    
    if text_input:
        st.subheader("🖼️ Sign Language Representation")
        text_to_sign_images(text_input)
        
        st.subheader("🔊 Speech Output")
        text_to_speech(text_input)

if __name__ == "__main__":
    main()
