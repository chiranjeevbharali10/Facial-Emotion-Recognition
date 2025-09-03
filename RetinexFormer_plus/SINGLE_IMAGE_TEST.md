# Single Image Testing with RetinexFormer

This guide shows you how to test RetinexFormer on your own low-light images without downloading any datasets.

## Quick Start

### 1. Download Pre-trained Weights
First, download the pre-trained weights from:
- [Baidu Disk](https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2) (code: `cyh2`) 
- [Google Drive](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV?usp=drive_link)

Place them in the `pretrained_weights/` folder.

### 2. Place Your Image
Put your low-light image in the project directory and name it `input.jpg` (or specify the path).

### 3. Run the Script
```bash
# Basic usage (uses LOL-v1 model)
python test_single_image.py

# With custom input/output
python test_single_image.py --input your_image.jpg --output enhanced.png

# Use different pre-trained model
python test_single_image.py --weights pretrained_weights/LOL_v2_real.pth --config Options/RetinexFormer_LOL_v2_real.yml

# Enable self-ensemble for better results (slower but better quality)
python test_single_image.py --self_ensemble

# Use specific GPU
python test_single_image.py --gpu 1
```

## Available Pre-trained Models

| Model | Config File | Weights File | Best For |
|-------|-------------|--------------|----------|
| LOL-v1 | `Options/RetinexFormer_LOL_v1.yml` | `LOL_v1.pth` | General low-light images |
| LOL-v2-real | `Options/RetinexFormer_LOL_v2_real.yml` | `LOL_v2_real.pth` | Real captured low-light |
| LOL-v2-synthetic | `Options/RetinexFormer_LOL_v2_synthetic.yml` | `LOL_v2_synthetic.pth` | Synthetic low-light |
| SID | `Options/RetinexFormer_SID.yml` | `SID.pth` | Very dark images |
| SMID | `Options/RetinexFormer_SMID.yml` | `SMID.pth` | Multi-illumination |
| SDSD-indoor | `Options/RetinexFormer_SDSD_indoor.yml` | `SDSD_indoor.pth` | Indoor low-light |
| SDSD-outdoor | `Options/RetinexFormer_SDSD_outdoor.yml` | `SDSD_outdoor.pth` | Outdoor low-light |
| FiveK | `Options/RetinexFormer_FiveK.yml` | `FiveK.pth` | Image enhancement |
| NTIRE | `Options/RetinexFormer_NTIRE.yml` | `NTIRE.pth` | Latest challenge dataset |

## Examples

```bash
# Test with LOL-v1 model (good for most cases)
python test_single_image.py --input dark_photo.jpg --output bright_photo.png

# Test with SID model (for very dark images)
python test_single_image.py --input very_dark.jpg --output enhanced.jpg --weights pretrained_weights/SID.pth --config Options/RetinexFormer_SID.yml

# Test with self-ensemble for best quality
python test_single_image.py --input night_photo.jpg --output enhanced_night.png --self_ensemble
```

## Tips

1. **Model Selection**: 
   - Use `LOL_v1` for general low-light images
   - Use `SID` for extremely dark images
   - Use `SDSD_indoor/outdoor` for specific environments

2. **Quality vs Speed**:
   - Without `--self_ensemble`: Faster, good quality
   - With `--self_ensemble`: Slower, better quality

3. **Image Formats**: Supports JPG, PNG, and other common formats

4. **Large Images**: The script automatically handles large images by splitting them

## Troubleshooting

- **CUDA out of memory**: Try using a smaller image or different GPU
- **Weights not found**: Make sure you downloaded the pre-trained weights
- **Input image not found**: Check the file path and make sure the image exists

## Requirements

Make sure you have installed the environment as described in the main README:
```bash
conda activate Retinexformer  # or your environment name
```

That's it! You can now enhance your low-light images without downloading any datasets. 