import os
from PIL import Image

# Assuming you have a dataset folder with images named as "A.jpg", "B.jpg", ..., "0.jpg", "1.jpg"
DATASET_PATH = "dataset/"

def text_to_sign(text):
    """Converts text to sign language images."""
    images = []
    for char in text.upper():
        img_path = os.path.join(DATASET_PATH, f"{char}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
    
    return images
