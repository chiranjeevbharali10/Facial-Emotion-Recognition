import os
from PIL import Image, ImageEnhance
import numpy as np
import random

# Set your input and output directories here
INPUT_DIR = 'archive/train/disgust'  # Change as needed
OUTPUT_DIR = 'augmented_train'  # Change as needed

# Only one augmentation per image now
TARGET_SIZE = (48, 48)  # Optional: resize after augmentation

# Augmentation functions
def random_rotate(img):
    angle = random.choice([0, 90, 180, 270])
    return img.rotate(angle)

def random_flip(img):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def random_brightness(img):
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(0.7, 1.3)
    return enhancer.enhance(factor)

def random_zoom(img):
    w, h = img.size
    zoom_factor = random.uniform(0.8, 1.0)
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    img = img.crop((left, top, left + new_w, top + new_h))
    return img.resize((w, h), Image.Resampling.LANCZOS)

def augment_image(img):
    # Apply a random sequence of augmentations
    img = random_rotate(img)
    img = random_flip(img)
    img = random_brightness(img)
    img = random_zoom(img)
    return img

def augment_and_save(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        save_dir = os.path.join(output_dir, rel_path)
        os.makedirs(save_dir, exist_ok=True)
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                try:
                    img = Image.open(input_path)
                    img = img.convert('L')  # Greyscale
                    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    base_name, ext = os.path.splitext(file)
                    # Save only one augmentation per image
                    aug_img = augment_image(img)
                    aug_img = aug_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    aug_img.save(os.path.join(save_dir, f'{base_name}_aug{ext}'))
                    print(f"Augmented: {input_path}")
                except Exception as e:
                    print(f"Failed to augment {input_path}: {e}")

if __name__ == '__main__':
    augment_and_save(INPUT_DIR, OUTPUT_DIR) 