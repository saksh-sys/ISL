import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
from gtts import gTTS
from PIL import Image

# Load the trained model
MODEL_PATH = "sign_model_mobilenetv2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Define label mapping (Ensure this matches your dataset)
label_map = {i: chr(65 + i) for i in range(26)}  # A-Z
label_map.update({26 + i: str(i + 1) for i in range(9)})  # 1-9

# Function to preprocess the hand image
def preprocess_hand(image):
    image_resized = cv2.resize(image, (128, 128)) / 255.0
    return np.expand_dims(image_resized, axis=0)

def main():
    st.title("🤟 Real-Time Sign to Text")

    # Text display area
    detected_text = st.text_area("Detected Signs", "", height=100, key="sign_text", disabled=True)

    # Clear text button
    if st.button("Clear Text"):
        st.session_state.sign_text = ""

    # Start camera
    cap = cv2.VideoCapture(0)
    
    last_capture_time = time.time()

    while cap.isOpened():
        current_time = time.time()
        
        # Capture frame every 1 second
        if current_time - last_capture_time >= 1:
            last_capture_time = current_time
            ret, frame = cap.read()

            if not ret:
                st.warning("⚠ Could not capture frame. Try again.")
                continue

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            detected_signs = []

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get bounding box of hand
                    x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                    y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
                    x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                    y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]

                    # Crop hand region
                    hand_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

                    if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                        hand_img = preprocess_hand(hand_img)

                        # Predict the sign
                        prediction = model.predict(hand_img)
                        predicted_index = np.argmax(prediction)
                        predicted_sign = label_map.get(predicted_index, "Unknown")

                        detected_signs.append(predicted_sign)

                # Append detected letters/numbers
                if detected_signs:
                    detected_text += "".join(detected_signs)
                    st.session_state.sign_text = detected_text
                    st.text_area("Detected Signs", detected_text, height=100, key="sign_text", disabled=True)

                    # Convert to Speech
                    tts = gTTS(text="".join(detected_signs), lang="en")
                    tts.save("speech_output.mp3")
                    st.audio("speech_output.mp3", format="audio/mp3", autoplay=True)

            # Display processed image with drawn landmarks
            st.image(frame, caption="Processed Frame", use_column_width=True)

if __name__ == "__main__":
    main()
