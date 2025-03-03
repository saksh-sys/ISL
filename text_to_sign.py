from PIL import Image, ImageDraw, ImageFont

def convert(text):
    """Generate an image representation of the given text."""
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((50, 40), text, fill=(0, 0, 0), font=font)

    return img
