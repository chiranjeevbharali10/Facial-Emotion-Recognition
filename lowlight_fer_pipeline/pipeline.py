import os
import sys
import glob
import argparse
from typing import Tuple, Optional, List

# Reduce TensorFlow/XLA log noise
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Keep TensorFlow/Keras separate from PyTorch parts
import tensorflow as tf
from tensorflow import keras

# --- RetinexFormer repo path resolution ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Allow explicit override via env var
ENV_RETINEX_ROOT = os.environ.get('RETINEX_ROOT')
DEFAULT_RETINEX_ROOT = os.path.join(PROJECT_ROOT, 'Retinexformer_plus-master')

# Placeholder; will be finalized in main() after CLI parsing if provided
RETINEX_ROOT: Optional[str] = ENV_RETINEX_ROOT or DEFAULT_RETINEX_ROOT

# Defer importing basicsr until RETINEX_ROOT is finalized in main()
create_model = None  # type: ignore
parse = None  # type: ignore


ENHANCED_DIR = os.path.join(os.path.dirname(__file__), 'enhanced')
os.makedirs(ENHANCED_DIR, exist_ok=True)


def _find_first(patterns: list) -> Optional[str]:
    """Return the first existing path matching any of the glob patterns."""
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def _auto_find_retinex_root(start_dir: str) -> Optional[str]:
    """Attempt to locate a folder containing basicsr within the workspace tree."""
    # Search common layout first
    candidate = os.path.join(start_dir, 'Retinexformer_plus-master')
    if os.path.isdir(os.path.join(candidate, 'basicsr')):
        return candidate
    # Fallback: glob for any basicsr directory under start_dir
    found = _find_first([os.path.join(start_dir, '**', 'basicsr', '__init__.py')])
    if found:
        return os.path.abspath(os.path.join(os.path.dirname(found), os.pardir))
    return None


def _inject_and_import_basicsr(retinex_root: str):
    """Ensure local basicsr is imported instead of pip version."""
    if retinex_root not in sys.path:
        sys.path.insert(0, retinex_root)
    global create_model, parse  # type: ignore
    from basicsr.models import create_model as _create_model  # type: ignore
    from basicsr.utils.options import parse as _parse  # type: ignore
    create_model = _create_model
    parse = _parse


def _pad_to_multiple_of_4(t: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad BCHW tensor reflectively so H and W are multiples of 4."""
    b, c, h, w = t.shape
    factor = 4
    H = ((h + factor) // factor) * factor
    W = ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    if padh or padw:
        t = F.pad(t, (0, padw, 0, padh), mode='reflect')
    return t, (padh, padw)


def _unpad(t: torch.Tensor, padh: int, padw: int) -> torch.Tensor:
    if padh or padw:
        return t[..., : t.shape[-2] - padh if padh else t.shape[-2], : t.shape[-1] - padw if padw else t.shape[-1]]
    return t


def _resolve_device(device: Optional[str]) -> str:
    """Resolve 'auto' -> 'cuda' if available else 'cpu'."""
    if device is None or device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in ('cpu', 'cuda'):
        return 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        return 'cpu'
    return device


def load_retinex_model(
    config_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    device: Optional[str] = 'auto',
) -> nn.Module:
    """Load RetinexFormer generator (net_g) with weights.

    device: 'auto'|'cpu'|'cuda'. 'auto' prefers CUDA when available.
    """
    device = _resolve_device(device)

    # Default locations
    if config_path is None:
        config_path = _find_first([
            os.path.join(RETINEX_ROOT, 'Options', 'RetinexFormer_LOL_v1.yml'),
            os.path.join(RETINEX_ROOT, 'Options', 'RetinexFormer_FiveK.yml'),
        ])
        if config_path is None:
            raise FileNotFoundError('Could not locate a default RetinexFormer config .yml in Options/.')

    if weights_path is None:
        # Try to find net_g_17000.pth inside experiments/*/models/
        weights_path = _find_first([
            os.path.join(RETINEX_ROOT, 'experiments', '**', 'models', 'net_g_17000.pth'),
            os.path.join(RETINEX_ROOT, '**', 'net_g_17000.pth'),
        ])
        if weights_path is None:
            raise FileNotFoundError('Could not find net_g_17000.pth. Please provide the correct path.')

    opt = parse(config_path, is_train=False)
    # Set device controls in options
    if device == 'cpu':
        opt['num_gpu'] = 0
        opt['dist'] = False
    else:  # cuda
        opt['num_gpu'] = max(1, torch.cuda.device_count())
        opt['dist'] = False
    model_restoration = create_model(opt).net_g

    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location='cpu')
    try:
        model_restoration.load_state_dict(checkpoint['params'])
    except Exception:
        new_checkpoint = {}
        for k in checkpoint['params']:
            new_checkpoint['module.' + k] = checkpoint['params'][k]
        model_restoration.load_state_dict(new_checkpoint)

    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        model_restoration = model_restoration.cuda()
        # Wrap in DataParallel if multiple GPUs
        if torch.cuda.device_count() > 1:
            model_restoration = nn.DataParallel(model_restoration)
    else:
        model_restoration = model_restoration.cpu()

    model_restoration.eval()
    return model_restoration


def _keras_f1_score(y_true, y_pred):
    """Keras-compatible F1 metric implementation for loading compiled models."""
    y_pred_bin = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred_bin, tf.float32), axis=None)
    fp = tf.reduce_sum(tf.cast((1.0 - y_true) * y_pred_bin, tf.float32), axis=None)
    fn = tf.reduce_sum(tf.cast(y_true * (1.0 - y_pred_bin), tf.float32), axis=None)
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2.0 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return f1


def load_resnet_classifier(h5_path: Optional[str] = None, cpu_only: bool = True) -> keras.Model:
    """Load the Keras .h5 classifier model. Optionally disable GPU visibility.

    Handles custom objects like f1_score; falls back to compile=False if needed.
    """
    if cpu_only:
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass

    if h5_path is None:
        # Try exact requested name first, then common variants found in the repo
        h5_path = _find_first([
            os.path.join(PROJECT_ROOT, '**', 'resnetfinetuned.h5'),
            os.path.join(PROJECT_ROOT, '**', 'resnetfinetune.h5'),
            os.path.join(PROJECT_ROOT, '**', 'resnet50_affectnet.h5'),
        ])
        if h5_path is None:
            raise FileNotFoundError('Could not find resnetfinetuned.h5 (or resnetfinetune.h5). Please provide the correct path.')

    custom_objects = {'f1_score': _keras_f1_score}
    # First attempt: load with custom objects
    try:
        model = keras.models.load_model(h5_path, custom_objects=custom_objects)
        return model
    except Exception as e_primary:
        primary_err = e_primary
    # Second attempt: load without compiling
    try:
        model = keras.models.load_model(h5_path, compile=False)
        return model
    except Exception as e_secondary:
        secondary_err = e_secondary
    # Final attempt: apply compatibility fix using helper if available
    try:
        from test_hf_model import try_fix_and_load_h5  # type: ignore
        model = try_fix_and_load_h5(h5_path)
        return model
    except Exception as e_compat:
        raise RuntimeError(
            'Failed to load legacy Keras .h5 model even after applying compatibility fix.\n'
            f'Primary load_model(custom_objects) error: {primary_err}\n'
            f'Secondary load_model(compile=False) error: {secondary_err}\n'
            f'Compatibility fix error: {e_compat}'
        )


def _infer_keras_input_shape(model: keras.Model) -> Tuple[int, int, int]:
    """Infer expected (H, W, C) for a Keras model's first input."""
    shape = getattr(model, 'input_shape', None)
    if isinstance(shape, list):
        shape = shape[0]
    # shape like (None, H, W, C)
    if shape and len(shape) == 4:
        _, h, w, c = shape
        h = 224 if h is None else h
        w = 224 if w is None else w
        c = 3 if c is None else c
        return int(h), int(w), int(c)
    # Fallbacks
    return 224, 224, 3


def enhance_with_retinex(img_path: str, model_restoration: nn.Module) -> np.ndarray:
    """Enhance a low-light image using RetinexFormer.

    Steps:
    - Open image
    - torchvision transforms: resize(224x224), to tensor
    - Run through RetinexFormer
    - Convert to uint8 RGB numpy
    - Save into enhanced/ and return
    """
    # Open and transform
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # [0,1], CxHxW
    ])
    tensor_img = transform(img).unsqueeze(0)  # 1x3x224x224

    device = next(model_restoration.parameters()).device
    tensor_img = tensor_img.to(device)

    # Pad to multiple of 4 as model expects
    input_batch, (padh, padw) = _pad_to_multiple_of_4(tensor_img)

    with torch.inference_mode():
        restored = model_restoration(input_batch)

    # Remove padding, clamp, convert to numpy HWC
    restored = _unpad(restored, padh, padw)
    restored = torch.clamp(restored, 0, 1)
    restored_np = restored.detach().cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
    restored_np_uint8 = (restored_np * 255.0 + 0.5).astype(np.uint8)

    # Save enhanced image
    base = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(ENHANCED_DIR, f'{base}_enhanced.png')
    Image.fromarray(restored_np_uint8).save(save_path)

    return restored_np_uint8


def classify_expression(enhanced_img: np.ndarray, classifier: keras.Model) -> Tuple[int, np.ndarray]:
    """Classify an enhanced image with the Keras ResNet classifier.

    - Resize to model's expected (H, W)
    - Convert to RGB or grayscale depending on expected channels
    - Scale to [0,1], predict class probabilities and argmax
    """
    if enhanced_img.dtype != np.uint8:
        enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)

    # Determine expected input shape from the model
    exp_h, exp_w, exp_c = _infer_keras_input_shape(classifier)

    img = Image.fromarray(enhanced_img)
    if exp_c == 1:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    img = img.resize((exp_w, exp_h), Image.BILINEAR)

    arr = np.asarray(img).astype('float32') / 255.0
    if exp_c == 1:
        arr = np.expand_dims(arr, axis=-1)  # HxWx1
    x = np.expand_dims(arr, axis=0)  # 1xHxWxC

    probs = classifier.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def _parse_labels_arg(labels_arg: Optional[str], num_classes: int) -> List[str]:
    """Return list of label names matching num_classes.

    - If labels_arg provided (comma-separated), use it and validate length.
    - Else choose sensible defaults for 7 or 5 classes; otherwise generic names.
    """
    if labels_arg:
        labels = [s.strip() for s in labels_arg.split(',') if s.strip()]
        if len(labels) != num_classes:
            raise ValueError(f'--labels count ({len(labels)}) does not match model classes ({num_classes}).')
        return labels
    if num_classes == 7:
        return ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    if num_classes == 5:
        return ['anger', 'fear', 'happy', 'sad', 'surprise']
    return [f'class_{i}' for i in range(num_classes)]


def main():
    parser = argparse.ArgumentParser(description='Low-light FER pipeline: RetinexFormer -> ResNet classifier')
    parser.add_argument('--retinex_root', type=str, default=None, help='Path to Retinexformer_plus-master root (overrides env/default)')
    parser.add_argument('--retinex_config', type=str, default=None, help='Path to RetinexFormer YAML config')
    parser.add_argument('--retinex_weights', type=str, default=None, help='Path to net_g_17000.pth')
    parser.add_argument('--resnet_h5', type=str, default=None, help='Path to resnetfinetuned.h5 (or resnetfinetune.h5)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device for RetinexFormer (default: auto)')
    parser.add_argument('--labels', type=str, default=None, help='Comma-separated label names in model index order')
    parser.add_argument('--image', type=str, default=os.path.join(os.path.dirname(__file__), 'test.jpg'), help='Input image path')
    args = parser.parse_args()

    # Resolve RetinexFormer root
    global RETINEX_ROOT
    if args.retinex_root is not None:
        RETINEX_ROOT = args.retinex_root
    if not RETINEX_ROOT or not os.path.isdir(os.path.join(RETINEX_ROOT, 'basicsr')):
        auto = _auto_find_retinex_root(PROJECT_ROOT)
        if auto:
            RETINEX_ROOT = auto
    if not RETINEX_ROOT or not os.path.isdir(os.path.join(RETINEX_ROOT, 'basicsr')):
        raise ImportError('Could not locate local RetinexFormer repo (basicsr). Use --retinex_root or set RETINEX_ROOT env var.')

    # Import local basicsr (with create_model available)
    _inject_and_import_basicsr(RETINEX_ROOT)

    # Resolve device and load models
    resolved_device = _resolve_device(args.device)
    retinex = load_retinex_model(args.retinex_config, args.retinex_weights, device=resolved_device)
    resnet = load_resnet_classifier(args.resnet_h5, cpu_only=(resolved_device == 'cpu'))

    # Run pipeline on sample image
    if not os.path.exists(args.image):
        print(f"Input image not found: {args.image}")
        print('Place test.jpg in the pipeline folder or pass --image path.')
        return

    enhanced = enhance_with_retinex(args.image, retinex)
    pred_idx, probs = classify_expression(enhanced, resnet)

    # Derive and print human-readable label
    labels = _parse_labels_arg(args.labels, len(probs))
    pred_label = labels[pred_idx]

    # Print result
    print(f'Retinex root: {RETINEX_ROOT}')
    print(f'Device: {resolved_device} (CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0})')
    print(f'Predicted expression: {pred_label} (index {pred_idx})')
    print('Probabilities:')
    for i, p in enumerate(probs):
        print(f'  {i}: {labels[i]} = {p:.4f}')


if __name__ == '__main__':
    main() 