import streamlit as st
import cv2
import numpy as np
import os

# Load pre-trained sign recognition model
MODEL_PATH = "sign_model_mobilenetv2.h5"  # Update with actual model path
DATASET_PATH = "dataset"

def load_model():
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)

def predict_sign(image, model):
    processed_img = cv2.resize(image, (64, 64)) / 255.0
    processed_img = np.expand_dims(processed_img, axis=0)
    prediction = model.predict(processed_img)
    class_labels = sorted(os.listdir(DATASET_PATH))
    return class_labels[np.argmax(prediction)]

def main():
    st.title("🤟 Sign to Text")
    model = load_model()
    
    camera = st.camera_input("📷 Capture Hand Sign")
    if camera:
        image = cv2.imdecode(np.frombuffer(camera.read(), np.uint8), 1)
        sign_text = predict_sign(image, model)
        st.success(f"Predicted Sign: {sign_text}")

if __name__ == "__main__":
    main()
