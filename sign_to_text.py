import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PIL import Image
import time

# Load models
model1 = tf.keras.models.load_model("sign_model.h5")
model2 = tf.keras.models.load_model("sign_model_mobilenetv2.h5")

# Define label mappings
labels = [str(i) for i in range(1, 10)] + [chr(i) for i in range(65, 91)]  # 1-9 and A-Z
label_dict = {i: label for i, label in enumerate(labels)}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def main():
    st.title("ISL Sign Language to Text")
    st.markdown("### Automatically captures hand signs and converts to text")

    # Read-only textbox for detected text
    detected_text = st.text_area("Detected Text:", "", height=100, key="text_box", disabled=True)
    clear_button = st.button("Clear Text")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    time.sleep(2)

    # Function to preprocess the image
    def preprocess_image(image):
        image = cv2.resize(image, (64, 64))
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)

    if cap.isOpened():
        stframe = st.empty()
        detected_chars = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                    
                    hand_img = rgb_frame[y_min:y_max, x_min:x_max]
                    if hand_img.size > 0:
                        hand_img = preprocess_image(hand_img)
                        
                        pred1 = model1.predict(hand_img)
                        pred2 = model2.predict(hand_img)
                        
                        label_index = np.argmax((pred1 + pred2) / 2)
                        predicted_char = label_dict.get(label_index, "")
                        
                        if predicted_char and (not detected_chars or predicted_char != detected_chars[-1]):
                            detected_chars.append(predicted_char)
                            st.session_state.text_box = "".join(detected_chars)
                    
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            stframe.image(frame, channels="RGB")
            
            if clear_button:
                detected_chars.clear()
                st.session_state.text_box = ""
                
        cap.release()
    else:
        st.error("Failed to open the camera")

if __name__ == "__main__":
    main()
