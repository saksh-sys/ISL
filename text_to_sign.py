import streamlit as st
import speech_recognition as sr
import time
from PIL import Image
import os

# Ensure Kaggle API is installed
os.system("pip install kaggle")

# Download and extract dataset if not already present
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.system("kaggle datasets download -d atharvadumbre/indian-sign-language-islrtc-referred --unzip -p dataset")

# Now you can use dataset_path for model training or inference

def load_sign_images():
    """Load ISL images from the dataset folder."""
    sign_images = {}
    dataset_path = "dataset"  # Adjust the path as per your dataset folder
    
    for letter in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
        letter_path = os.path.join(dataset_path, letter)
        if os.path.exists(letter_path):
            images = [os.path.join(letter_path, img) for img in os.listdir(letter_path) if img.endswith(".jpg")]
            sign_images[letter] = images[0] if images else None  # Take the first image
    
    return sign_images

def text_to_sign(text, sign_images):
    """Display sign language images corresponding to the text."""
    st.write("### ISL Representation:")
    col1, col2, col3 = st.columns(3)
    
    for i, char in enumerate(text.upper()):
        if char in sign_images and sign_images[char]:
            img = Image.open(sign_images[char])
            with [col1, col2, col3][i % 3]:
                st.image(img, caption=char, use_column_width=True)
            time.sleep(0.5)  # Delay for effect

def recognize_speech():
    """Convert speech to text using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Speak now...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
        except sr.RequestError:
            st.error("Could not request results. Check your internet connection.")
        except Exception as e:
            st.error(f"Error: {e}")
    return ""

def main():
    st.title("📖 Text to Sign Language Converter")
    sign_images = load_sign_images()
    
    input_type = st.radio("Choose Input Type:", ["📝 Text", "🎤 Speech"])
    
    if input_type == "📝 Text":
        user_input = st.text_input("Enter text (A-Z, 0-9 only):")
        if st.button("Convert to Sign Language") and user_input:
            text_to_sign(user_input, sign_images)
    
    elif input_type == "🎤 Speech":
        if st.button("Start Listening"):
            recognized_text = recognize_speech()
            if recognized_text:
                text_to_sign(recognized_text, sign_images)

if __name__ == "__main__":
    main()
