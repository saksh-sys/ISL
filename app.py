import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
from PIL import Image
import tempfile

# Load the trained model
model = tf.keras.models.load_model("sign_model_mobilenetv2.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# UI Title
st.title("Indian Sign Language Translator")
st.write("Upload an image or use your webcam to recognize Indian Sign Language gestures.")

# Function to preprocess the image
def preprocess_hand(image):
    image_resized = cv2.resize(image, (128, 128)) / 255.0
    return np.expand_dims(image_resized, axis=0)

# Function to predict sign
def predict_sign(image):
    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    return predicted_index

# Class Labels (Update as per your dataset)
label_map = {0: "Hello", 1: "Thank You", 2: "Yes", 3: "No", 4: "I Love You"}

# Upload Image
uploaded_file = st.file_uploader("Upload an image of a hand gesture", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and Predict
    processed_image = preprocess_hand(image)
    predicted_label = predict_sign(processed_image)

    # Display Result
    result_text = label_map.get(predicted_label, "Unknown Sign")
    st.subheader(f"Predicted Sign: {result_text}")

    # Convert to Speech
    tts = gTTS(text=result_text, lang="en")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    st.audio(temp_audio.name, format="audio/mp3")
