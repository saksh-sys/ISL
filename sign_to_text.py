import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("model_sign_99.h5")

# Define class labels (update based on dataset)
class_labels = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]  # A-Z + 0-9

def predict_sign(img_path):
    """Loads an image and predicts the corresponding sign."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    return predicted_class
