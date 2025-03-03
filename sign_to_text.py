import streamlit as st
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PIL import Image
import imageio
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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

class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.detected_chars = []
    
    def preprocess_image(self, image):
        image = image.resize((64, 64))
        image = np.array(image).astype("float32") / 255.0
        return np.expand_dims(image, axis=0)
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * img.shape[1])
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * img.shape[0])
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * img.shape[1])
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * img.shape[0])
                
                if x_max > x_min and y_max > y_min:
                    hand_img = rgb_frame[y_min:y_max, x_min:x_max]
                    hand_img = Image.fromarray(hand_img)
                    hand_img = self.preprocess_image(hand_img)
                    
                    pred1 = model1.predict(hand_img)
                    pred2 = model2.predict(hand_img)
                    
                    label_index = np.argmax((pred1 + pred2) / 2)
                    predicted_char = label_dict.get(label_index, "")
                    
                    if predicted_char and (not self.detected_chars or predicted_char != self.detected_chars[-1]):
                        self.detected_chars.append(predicted_char)
                        st.session_state.text_box = "".join(self.detected_chars)
        
        return frame

def main():
    st.title("ISL Sign Language to Text")
    st.markdown("### Automatically captures hand signs and converts to text")
    
    # Read-only textbox for detected text
    st.text_area("Detected Text:", "", height=100, key="text_box", disabled=True)
    if st.button("Clear Text"):
        st.session_state.text_box = ""
    
    webrtc_streamer(key="sign-detection", video_processor_factory=SignProcessor)

if __name__ == "__main__":
    main()
