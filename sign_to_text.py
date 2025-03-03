import streamlit as st
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PIL import Image
import time
import asyncio
import av
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Ensure an event loop exists
asyncio.set_event_loop(asyncio.new_event_loop())

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

def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

class HandSignProcessor(VideoProcessorBase):
    def __init__(self):
        self.detected_chars = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * image.shape[1])
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * image.shape[0])
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * image.shape[1])
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * image.shape[0])
                
                hand_img = rgb_frame[y_min:y_max, x_min:x_max]
                if hand_img.size > 0:
                    hand_img = Image.fromarray(hand_img)
                    hand_img = preprocess_image(hand_img)
                    
                    pred1 = model1.predict(hand_img)
                    pred2 = model2.predict(hand_img)
                    
                    label_index = np.argmax((pred1 + pred2) / 2)
                    predicted_char = label_dict.get(label_index, "")
                    
                    if predicted_char and (not self.detected_chars or predicted_char != self.detected_chars[-1]):
                        self.detected_chars.append(predicted_char)
                        st.session_state.text_box = "".join(self.detected_chars)
                
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")

def main():
    st.title("ISL Sign Language to Text")
    st.markdown("### Automatically captures hand signs and converts to text")

    # Read-only textbox for detected text
    st.text_area("Detected Text:", "", height=100, key="text_box", disabled=True)
    if st.button("Clear Text"):
        st.session_state.text_box = ""
    
    # WebRTC-based camera stream
    webrtc_streamer(key="sign-detection", video_processor_factory=HandSignProcessor, media_stream_constraints={"video": True, "audio": False})

if __name__ == "__main__":
    main()
