import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
from PIL import Image
import tempfile
import os

# Load the trained model
MODEL_PATH = "sign_model_mobilenetv2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define label mapping (Ensure this matches your dataset)
label_map = {i: chr(65 + i) for i in range(26)}  # A-Z
label_map.update({26 + i: str(i + 1) for i in range(9)})  # 1-9

# Function to preprocess the hand image
def preprocess_hand(image):
    image_resized = cv2.resize(image, (128, 128)) / 255.0
    return np.expand_dims(image_resized, axis=0)

def main():
    st.title("🤟 Sign to Text Conversion")

    # Camera Input
    st.write("### 📷 Capture Sign Language Gesture")
    captured_image = st.camera_input("Take a picture")

    if captured_image:
        # Convert to OpenCV format
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(captured_image.getvalue())
            temp_file_path = temp_file.name
        
        image = cv2.imread(temp_file_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe
        result = hands.process(rgb_image)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get bounding box of hand
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * image.shape[1]
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * image.shape[0]
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * image.shape[1]
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * image.shape[0]

                # Crop hand region
                hand_img = image[int(y_min):int(y_max), int(x_min):int(x_max)]
                
                if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                    hand_img = preprocess_hand(hand_img)

                    # Predict the sign
                    prediction = model.predict(hand_img)
                    predicted_index = np.argmax(prediction)
                    predicted_sign = label_map.get(predicted_index, "Unknown")

                    # Display results
                    st.image(image, caption=f"Processed Image", use_column_width=True)
                    st.write(f"### 🔠 Predicted Sign: **{predicted_sign}**")

                    # Convert to Speech
                    tts = gTTS(text=str(predicted_sign), lang="en")
                    speech_path = "speech_output.mp3"
                    tts.save(speech_path)
                    st.audio(speech_path, format="audio/mp3", autoplay=True)

        else:
            st.warning("⚠ No hand detected. Try again.")

if __name__ == "__main__": 
    main()   
