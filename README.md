
A unified pipeline for **Facial Emotion Recognition (FER)** in low-light conditions.  
This project integrates **RetinexFormer** (for image enhancement) with **ResNet50/VGG16**-based models for emotion classification.  

---

 🔹 Features
- ResNet50 & VGG16 for facial emotion recognition  
- RetinexFormer transformer-based low-light enhancement  
- Real-time webcam support  
- Pre-trained models for quick testing  
- Training with advanced augmentation & early stopping  

---

## 📂 Project Structure
```
IEEE_FINAL/
├── FER/
|   ├── Faces/            # Emotion Recognition
│   ├── Final_h5/          # Pre-trained models
│   ├── TESTING/           # Testing scripts
|   ├── dataAUG/            
│   └── resnet_train.py     # Training script
│
├── lowlight_fer_pipeline/  # Enhancement + Recognition Integration
│   └── pipeline.py
|   └── Test_image/
│
├── RetinexFormer_plus/     # Low-Light Enhancement
│   ├── basicsr/
│   ├── Enhancement/
│   └── webcam_retinexformer.py
│
├── requirements.txt
└── README.md


````

---

## ⚙️ Installation
```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: module-specific dependencies
pip install -r Facial/requirements.txt
pip install -r lowlight_fer_pipeline/requirements.txt
````

**Requirements:** Python 3.8+, TensorFlow ≥ 2.8, Torch ≥ 1.12, OpenCV ≥ 4.5, CUDA GPU recommended

---

## 🚀 Usage

### 1. Emotion Recognition

```bash
cd Facial/TESTING
python predict_image.py --image_path ../Faces/happy.jpg
```

### 2. Low-Light Enhancement

```bash
cd RetinexFormer_plus
python test_single_image.py --input_path dark.jpg --output_path enhanced.jpg
```

### 3. Integrated Pipeline

```bash
cd lowlight_fer_pipeline
python pipeline.py --image image_path
```

### 4. Real-Time

```bash
# Live emotion recognition
cd Facial/TESTING
python webcam_emotion_predict.py

# Live enhancement
cd RetinexFormer_plus
python webcam_retinexformer.py
```

---

## 🏋️ Training

**Train FER model (ResNet50/VGG16):**

```bash
cd Facial
python resnet_train.py
```

**Train RetinexFormer:**

```bash
cd RetinexFormer_plus
python basicsr/train.py -opt Options/RetinexFormer_LOL_v1.yml
```

---

## 📊 Model Info

**Emotion Recognition**

* Classes: Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral
* Input size: 224×224×3
* Optimizer: Adam + LR scheduling
* Features: Data augmentation, early stopping, checkpointing

**Low-Light Enhancement**

* Model: RetinexFormer (ICCV 2023)
* Datasets: LOL, SID, SMID
* Supports up to 4000×6000 resolution

---

## 🐛 Troubleshooting

* **CUDA OOM** → reduce batch size
* **Import errors** → reinstall dependencies
* **Model loading issues** → check `.h5` / checkpoint paths

---

## 📚 References

* RetinexFormer: *ICCV 2023*
* ResNet: *CVPR 2016*
* VGG: *ICLR 2015*
* Datasets: FER2013, LOL, SID

---

## 📜 License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

* RetinexFormer & BasicSR teams
* TensorFlow, PyTorch, and OpenCV communities

```


