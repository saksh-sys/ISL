import os
from PIL import Image

# Path to dataset
DATASET_PATH = "dataset/"

def text_to_sign(text):
    """Converts text to corresponding sign images."""
    images = []
    for char in text.upper():
        folder_path = os.path.join(DATASET_PATH, char)  # Navigate into subfolder
        img_path = os.path.join(folder_path, "1.jpg")  # Fetch "1.jpg" inside folder

        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)

    return images
