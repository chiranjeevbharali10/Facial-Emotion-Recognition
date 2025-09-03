#!/usr/bin/env python3
"""
Simple script to test RetinexFormer on a single image without BasicSR framework.
This bypasses CUDA issues and works directly with the model.
"""

import numpy as np
import os
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import img_as_ubyte

# Import the RetinexFormer model directly
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer

def load_img(path):
    """Load image from path"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_img(path, img):
    """Save image to path"""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def main():
    parser = argparse.ArgumentParser(description='Test RetinexFormer on a single image')
    parser.add_argument('--input', type=str, default='input.jpg', 
                       help='Input image path (default: input.jpg)')
    parser.add_argument('--output', type=str, default='output.png', 
                       help='Output image path (default: output.png)')
    parser.add_argument('--weights', type=str, default='pretrained_weights/LOL_v1.pth',
                       help='Path to model weights')
    
    args = parser.parse_args()
    
    # Check if input image exists
    if not os.path.exists(args.input):
        print(f"Error: Input image '{args.input}' not found!")
        print("Please place your low-light image in the current directory or specify the correct path.")
        return
    
    # Check if weights exist
    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found!")
        print("Please download the pre-trained weights first.")
        return
    
    # Create model directly
    print("Creating RetinexFormer model...")
    model = RetinexFormer(
        in_channels=3, 
        out_channels=3, 
        n_feat=40, 
        stage=1, 
        num_blocks=[1,2,2]
    )
    
    # Load weights
    print(f"Loading weights from: {args.weights}")
    checkpoint = torch.load(args.weights, map_location='cpu')
    
    try:
        model.load_state_dict(checkpoint['params'])
    except:
        # Try with module prefix
        new_checkpoint = {}
        for k in checkpoint['params']:
            if k.startswith('module.'):
                new_checkpoint[k[7:]] = checkpoint['params'][k]  # Remove 'module.' prefix
            else:
                new_checkpoint[k] = checkpoint['params'][k]
        model.load_state_dict(new_checkpoint)
    
    # Move to CPU (since CUDA is not available)
    model = model.cpu()
    model.eval()
    
    # Load and process image
    print(f"Processing image: {args.input}")
    img = np.float32(load_img(args.input)) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cpu()
    
    # Padding for images not multiples of 4
    factor = 4
    b, c, h, w = img.shape
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    img = F.pad(img, (0, padw, 0, padh), 'reflect')
    
    # Process image
    print("Running inference...")
    with torch.inference_mode():
        if h < 3000 and w < 3000:
            restored = model(img)
        else:
            # Split and test for very large images
            img_1 = img[:, :, :, 1::2]
            img_2 = img[:, :, :, 0::2]
            restored_1 = model(img_1)
            restored_2 = model(img_2)
            restored = torch.zeros_like(img)
            restored[:, :, :, 1::2] = restored_1
            restored[:, :, :, 0::2] = restored_2
    
    # Unpad and convert to numpy
    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0, 1).detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    
    # Save result
    save_img(args.output, img_as_ubyte(restored))
    print(f"Enhanced image saved to: {args.output}")
    print("Done!")

if __name__ == '__main__':
    main() 