import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer

# --------- CONFIG ---------
MODEL_PATH = 'experiments/RetinexFormer_LOL_v1_gpu/models/net_g_11000.pth'  # Change if needed
N_FEAT = 40
STAGE = 1
NUM_BLOCKS = [1, 2, 2]

# --------------------------

def preprocess_frame(frame, device):
    # Convert BGR to RGB, normalize, to tensor, add batch dim
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    # Pad to multiple of 4
    factor = 4
    b, c, h, w = img.shape
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    img = F.pad(img, (0, padw, 0, padh), 'reflect')
    return img, h, w

def postprocess_tensor(tensor, h, w):
    # Remove padding, clamp, convert to uint8 numpy RGB
    img = tensor[:, :, :h, :w].squeeze(0).detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # CHW to HWC
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def load_retinexformer(model_path, device):
    model = RetinexFormer(
        in_channels=3,
        out_channels=3,
        n_feat=N_FEAT,
        stage=STAGE,
        num_blocks=NUM_BLOCKS
    )
    checkpoint = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['params'])
    except:
        # Remove 'module.' prefix if present
        new_checkpoint = {}
        for k in checkpoint['params']:
            if k.startswith('module.'):
                new_checkpoint[k[7:]] = checkpoint['params'][k]
            else:
                new_checkpoint[k] = checkpoint['params'][k]
        model.load_state_dict(new_checkpoint)
    model = model.to(device)
    model.eval()
    return model

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = load_retinexformer(MODEL_PATH, device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    print("Press 'q' to quit.")
    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            img_tensor, h, w = preprocess_frame(frame, device)
            enhanced = model(img_tensor)
            enhanced_img = postprocess_tensor(enhanced, h, w)
            # Show original and enhanced side by side
            combined = np.hstack((frame, enhanced_img))
            cv2.imshow('Original (left) | Enhanced (right)', combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 