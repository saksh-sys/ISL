import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
from PIL import Image
import sign_to_text  # Import sign to text module
import text_to_sign  # Import text to sign module

# GitHub Raw URL for model
MODEL_URL = "https://github.com/saksh-sys/ISL/blob/main/model_sign_99.h5"

@st.cache_resource
def load_model():
    """Download and load the model from GitHub."""
    response = requests.get(MODEL_URL)
    model_path = "model_sign_99.h5"
    with open(model_path, "wb") as f:
        f.write(response.content)
    return tf.keras.models.load_model(model_path)

# Load Model
model = load_model()

# Class labels (Adjust based on your dataset)
class_names = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]  # A-Z + 0-9

st.title("Indian Sign Language Recognition")
st.write("Upload an image to classify the sign.")

# Upload Image
uploaded_file = st.file_uploader("Choose a sign language image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Preprocess Image
    img = image_data.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make Prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"**Predicted Sign:** {predicted_class}")

    # Convert Sign to Text
    sign_text = sign_to_text.convert(predicted_class)
    st.write(f"**Sign to Text:** {sign_text}")

    # Convert Text to Sign
    text_sign = text_to_sign.convert(sign_text)
    st.image(text_sign, caption="Generated Sign Image")

