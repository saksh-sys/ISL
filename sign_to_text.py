import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import mediapipe as mp

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

def main():
    st.title("ISL Sign Language to Text")
    st.markdown("### Automatically captures hand signs and converts to text")

    # Read-only textbox for detected text
    detected_text = st.empty()
    if "text_box" not in st.session_state:
        st.session_state.text_box = ""

    # Access the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to open the camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract bounding box
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                # Ensure the bounding box is within the frame
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                # Extract hand image
                hand_img = rgb_frame[y_min:y_max, x_min:x_max]
                if hand_img.size > 0:
                    hand_img = Image.fromarray(hand_img)
                    hand_img = preprocess_image(hand_img)

                    # Predict with both models and average the results
                    pred1 = model1.predict(hand_img)
                    pred2 = model2.predict(hand_img)
                    avg_pred = (pred1 + pred2) / 2

                    label_index = np.argmax(avg_pred)
                    predicted_char = label_dict.get(label_index, "")

                    if predicted_char:
                        st.session_state.text_box += predicted_char

        # Display the frame
        st.image(frame, channels="BGR")

        # Update detected text
        detected_text.text_area("Detected Text:", st.session_state.text_box, height=100, disabled=True)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
