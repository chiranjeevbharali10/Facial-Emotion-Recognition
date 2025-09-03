#!/usr/bin/env python3
"""
Script to help download pre-trained weights for RetinexFormer.
"""

import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def main():
    print("="*60)
    print("RETINEXFORMER WEIGHTS DOWNLOADER")
    print("="*60)
    print()
    print("Since direct download links are not available, please:")
    print()
    print("1. Download LOL_v1.pth from one of these sources:")
    print("   - Baidu Disk: https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2")
    print("   - Google Drive: https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV?usp=drive_link")
    print()
    print("2. Place the downloaded LOL_v1.pth file in the pretrained_weights/ folder")
    print()
    print("3. Then run: python simple_test.py --input your_image.jpg --output enhanced.png")
    print()
    print("="*60)
    
    # Check if weights folder exists
    if not os.path.exists('pretrained_weights'):
        os.makedirs('pretrained_weights')
        print("Created pretrained_weights/ folder")
    
    # Check if weights file exists
    if os.path.exists('pretrained_weights/LOL_v1.pth'):
        print("✅ LOL_v1.pth found! You can now run the test.")
    else:
        print("❌ LOL_v1.pth not found. Please download it first.")

if __name__ == '__main__':
    main() 