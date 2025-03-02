import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the trained model
MODEL_PATH = "sign_model_mobilenetv2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define label mapping (A-Z and 1-9)
label_map = {i: chr(65 + i) for i in range(26)}  # A-Z
label_map.update({26 + i: str(i + 1) for i in range(9)})  # 1-9

# **Ensure session state is initialized properly**
if "detected_text" not in st.session_state:
    st.session_state["detected_text"] = ""

# Function to preprocess the hand image
def preprocess_hand(image):
    image_resized = cv2.resize(image, (128, 128)) / 255.0
    return np.expand_dims(image_resized, axis=0)

def detect_sign():
    cap = cv2.VideoCapture(0)  # Open webcam
    stframe = st.empty()  # Placeholder for webcam feed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠ Could not access the camera.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get bounding box
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]

                # Crop and preprocess the hand region
                hand_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                
                if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                    hand_img = preprocess_hand(hand_img)

                    # Predict the sign
                    prediction = model.predict(hand_img)
                    predicted_index = np.argmax(prediction)
                    predicted_sign = label_map.get(predicted_index, "")

                    # Append detected sign to text
                    if predicted_sign:
                        st.session_state["detected_text"] += predicted_sign

        # Display camera feed in Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

def main():
    st.title("🤟 Real-Time Sign to Text Conversion")

    # **Ensure session state is initialized before use**
    if "detected_text" not in st.session_state:
        st.session_state["detected_text"] = ""

    # Webcam Start Button
    if st.button("Start Camera"):
        detect_sign()

    # Display Detected Text
    st.text_area("Detected Text:", value=st.session_state["detected_text"], height=100, disabled=True)

    # Clear Text Button
    if st.button("Clear Text"):
        st.session_state["detected_text"] = ""

if __name__ == "__main__":
    main()
