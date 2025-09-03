import os
from PIL import Image

# Set your input and output directories here
INPUT_DIR = 'conversion'  # Change as needed
OUTPUT_DIR = 'HELLO'  # Change as needed

TARGET_SIZE = (48, 48)


def convert_and_save(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        # Compute relative path to preserve subfolder structure
        rel_path = os.path.relpath(root, input_dir)
        save_dir = os.path.join(output_dir, rel_path)
        os.makedirs(save_dir, exist_ok=True)
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(save_dir, file)
                try:
                    img = Image.open(input_path)
                    img = img.convert('L')  # Convert to greyscale
                    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    img.save(output_path)
                    print(f"Saved: {output_path}")
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")

if __name__ == '__main__':
    convert_and_save(INPUT_DIR, OUTPUT_DIR) 