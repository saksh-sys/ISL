import os
from PIL import Image

# Path to dataset
DATASET_PATH = "dataset/"

def text_to_sign(text):
    """Converts text to corresponding sign images."""
    images = []
    for char in text.upper():
        img_path = os.path.join(DATASET_PATH, f"{char}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
    
    return images
