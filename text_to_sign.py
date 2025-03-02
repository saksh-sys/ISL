import streamlit as st
import os
from PIL import Image

# Path to ISL images (ensure images are stored correctly)
ISL_IMAGE_PATH = "sign_images/"

def load_sign_images(text):
    signs = []
    for char in text.upper():
        if char.isalnum():  # Only allow A-Z and 1-9
            image_path = os.path.join(ISL_IMAGE_PATH, f"{char}.png")
            if os.path.exists(image_path):
                signs.append((char, Image.open(image_path)))
    return signs

def main():
    st.title("📖 Text to Sign Language")

    text_input = st.text_input("Enter text (A-Z, 1-9 only):")
    
    if text_input:
        signs = load_sign_images(text_input)
        if signs:
            st.write("🔤 **Corresponding Sign Language Representation:**")
            cols = st.columns(len(signs))  # Create columns for images
            for col, (char, img) in zip(cols, signs):
                col.image(img, caption=char, use_column_width=True)
        else:
            st.warning("⚠ No matching signs found!")

if __name__ == "__main__":
    main()
