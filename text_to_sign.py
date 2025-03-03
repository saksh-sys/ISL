import os
import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
from io import BytesIO

# Define dataset path (update if needed)
DATASET_PATH = "dataset"

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
    st.title("📖 Text to Sign Language")
    
    text_input = st.text_input("Enter text:")
    
    if text_input:
        st.subheader("🖼️ Sign Language Representation")
        text_to_sign_images(text_input.upper())
        
        st.subheader("🔊 Speech Output")
        text_to_speech(text_input)

if __name__ == "__main__":
    main()
