#!/usr/bin/env python3
"""
Helper script to set up single image testing with data_single folder structure.
This creates the minimal dataset structure needed for testing.
"""

import os
import shutil
import argparse

def setup_single_test():
    """Set up the data_single folder structure for LOLv1 testing"""
    
    # Create the folder structure
    folders = [
        'data_single/LOLv1/Test/input',
        'data_single/LOLv1/Test/target'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")
    
    print("\n" + "="*50)
    print("SINGLE IMAGE TESTING SETUP COMPLETE!")
    print("="*50)
    print("\nTo test your images:")
    print("1. Place your low-light image in: data_single/LOLv1/Test/input/")
    print("2. If you have ground truth, place it in: data_single/LOLv1/Test/target/")
    print("3. Run the test script:")
    print("   python Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1_single.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1")
    print("\nOr use the simple single image script:")
    print("   python test_single_image.py --input your_image.jpg --output enhanced.png")
    print("\n" + "="*50)

def copy_image_to_input(image_path):
    """Copy an image to the input folder"""
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found!")
        return False
    
    input_folder = 'data_single/LOLv1/Test/input'
    filename = os.path.basename(image_path)
    destination = os.path.join(input_folder, filename)
    
    shutil.copy2(image_path, destination)
    print(f"Copied {image_path} to {destination}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Set up single image testing')
    parser.add_argument('--image', type=str, help='Path to image to copy to input folder')
    
    args = parser.parse_args()
    
    # Set up folder structure
    setup_single_test()
    
    # Copy image if provided
    if args.image:
        copy_image_to_input(args.image)

if __name__ == '__main__':
    main() 