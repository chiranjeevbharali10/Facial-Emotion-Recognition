#!/usr/bin/env python3
"""
Script to test the trained Retinexformer model on your own sample image.
"""

import numpy as np
import os
import argparse
import cv2
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte

# Import the RetinexFormer model
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer

def load_img(path):
    """Load image from path"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_img(path, img):
    """Save image to path"""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def main():
    parser = argparse.ArgumentParser(description='Test trained Retinexformer on your own image')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input image path (your low-light image)')
    parser.add_argument('--output', type=str, default='enhanced.png', 
                       help='Output image path (default: enhanced.png)')
    parser.add_argument('--model', type=str, default='experiments/RetinexFormer_LOL_v1_gpu/models/net_g_11000.pth',
                       help='Path to trained model (default: latest model)')
    
    args = parser.parse_args()
    
    # Check if input image exists
    if not os.path.exists(args.input):
        print(f"Error: Input image '{args.input}' not found!")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Available models:")
        model_dir = "experiments/RetinexFormer_LOL_v1_gpu/models"
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.pth'):
                    print(f"  - {model_dir}/{file}")
        return
    
    # Create model
    print("Creating RetinexFormer model...")
    model = RetinexFormer(
        in_channels=3, 
        out_channels=3, 
        n_feat=40, 
        stage=1, 
        num_blocks=[1,2,2]
    )
    
    # Load trained weights
    print(f"Loading weights from: {args.model}")
    checkpoint = torch.load(args.model, map_location='cpu')
    
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
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load and process image
    print(f"Processing image: {args.input}")
    img = np.float32(load_img(args.input)) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    
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
    restored = torch.clamp(restored, 0, 1).detach().permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
    
    # Save result
    save_img(args.output, img_as_ubyte(restored))
    print(f"Enhanced image saved to: {args.output}")
    print("Done!")

if __name__ == '__main__':
    main() 