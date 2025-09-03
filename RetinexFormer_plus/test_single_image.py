#!/usr/bin/env python3
"""
Simple script to test RetinexFormer on a single image without needing datasets.
Just place your low-light image in the same directory and run this script.
"""

import numpy as np
import os
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import img_as_ubyte
import yaml

from basicsr.models import create_model
from basicsr.utils.options import parse

def self_ensemble(x, model):
    """Self-ensemble for better results"""
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)

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
    parser.add_argument('--config', type=str, default='Options/RetinexFormer_LOL_v1.yml',
                       help='Path to config file')
    parser.add_argument('--self_ensemble', action='store_true', 
                       help='Use self-ensemble for better results')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f'Using GPU: {args.gpu}')
    else:
        print('Using CPU (CUDA not available or --cpu flag used)')
    
    # Check if input image exists
    if not os.path.exists(args.input):
        print(f"Error: Input image '{args.input}' not found!")
        print("Please place your low-light image in the current directory or specify the correct path.")
        return
    
    # Load configuration
    opt = parse(args.config, is_train=False)
    opt['dist'] = False
    
    # Create model
    model_restoration = create_model(opt).net_g
    
    # Load weights
    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found!")
        print("Please download the pre-trained weights first.")
        return
    
    checkpoint = torch.load(args.weights, map_location='cpu')
    try:
        model_restoration.load_state_dict(checkpoint['params'])
    except:
        new_checkpoint = {}
        for k in checkpoint['params']:
            new_checkpoint['module.' + k] = checkpoint['params'][k]
        model_restoration.load_state_dict(new_checkpoint)
    
    print(f"Loaded weights: {args.weights}")
    
    # Move model to device
    if use_cuda:
        model_restoration.cuda()
        model_restoration = nn.DataParallel(model_restoration)
    else:
        model_restoration = model_restoration.cpu()
    
    model_restoration.eval()
    
    # Load and process image
    print(f"Processing image: {args.input}")
    img = np.float32(load_img(args.input)) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)
    if use_cuda:
        input_ = img.unsqueeze(0).cuda()
    else:
        input_ = img.unsqueeze(0).cpu()
    
    # Padding for images not multiples of 4
    factor = 4
    b, c, h, w = input_.shape
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    
    # Process image
    with torch.inference_mode():
        if h < 3000 and w < 3000:
            if args.self_ensemble:
                restored = self_ensemble(input_, model_restoration)
            else:
                restored = model_restoration(input_)
        else:
            # Split and test for very large images
            input_1 = input_[:, :, :, 1::2]
            input_2 = input_[:, :, :, 0::2]
            if args.self_ensemble:
                restored_1 = self_ensemble(input_1, model_restoration)
                restored_2 = self_ensemble(input_2, model_restoration)
            else:
                restored_1 = model_restoration(input_1)
                restored_2 = model_restoration(input_2)
            restored = torch.zeros_like(input_)
            restored[:, :, :, 1::2] = restored_1
            restored[:, :, :, 0::2] = restored_2
    
    # Unpad and convert to numpy
    restored = restored[:, :, :h, :w]
    if use_cuda:
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    else:
        restored = torch.clamp(restored, 0, 1).detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    
    # Save result
    save_img(args.output, img_as_ubyte(restored))
    print(f"Enhanced image saved to: {args.output}")
    print("Done!")

if __name__ == '__main__':
    main() 