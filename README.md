# 📦 CIFAR-10 Classification using Custom ResNet (PyTorch + HuggingFace)

This project implements a **custom ResNet model** trained on the **CIFAR-10 dataset**, loaded directly from HuggingFace (`uoft-cs/cifar10`).  
The model is built from scratch using Residual Blocks, fully trained, evaluated, and tested on external real-world images.

---

## 📌 Project Overview

The goal of this project is to classify images from CIFAR-10 into **10 object categories** using a custom deep convolutional neural network.

Key components of the project include:

- Loading CIFAR-10 from HuggingFace Datasets  
- Applying grayscale conversion, normalization, and resizing  
- Building a simplified **ResNet-style model** from scratch  
- Training and evaluating accuracy  
- Testing the model on a folder of real images  
- Plotting training loss and accuracy curves  
- Saving the trained model for deployment

---

## 🧠 CIFAR-10 Classes

The 10 categories:

These are further used during custom folder-based testing.

---

## 🔄 Preprocessing

Each image is processed using:

- Grayscale conversion (`1-channel`)
- Resize to **32×32**
- Convert to PyTorch tensor
- Normalize to mean=0.5, std=0.5

The preprocessing is applied using HuggingFace's `.with_transform()`.

---

## 🧱 Model Architecture — Custom ResNet

The model mimics a lightweight ResNet-like design:

### **ResidualBlock**
- Two 3×3 convolutions  
- BatchNorm  
- Optional 1×1 convolution shortcut  
- Activation: ReLU  

### **ResNet Backbone**
- Initial 7×7 convolution + max pool  
- Residual Layer 1 (64 channels)  
- Residual Layer 2 (128 channels)  
- Residual Layer 3 (256 channels)  
- AdaptiveAvgPool → Fully Connected Layer  
- Final output: **10 classes**

This architecture is powerful enough for CIFAR-10 classification while staying compact.

---

## 🏋️ Training Setup

- **Optimizer:** Adam  
- **Loss:** CrossEntropyLoss  
- **Batch Size:** 32  
- **Device:** MPS (Apple Silicon) or CPU  
- **Training Loop:** Records  
  - `loss per epoch`  
  - custom folder test accuracy  
  - CIFAR test set accuracy  

Training ends early if custom accuracy exceeds 85%.

---

## 🧪 Evaluation

Two types of evaluation are used:

### ✔️ 1. Test DataLoader Accuracy  
Direct inference on CIFAR-10 test split.

### ✔️ 2. External Folder Testing  
Images inside a folder (e.g., `CIFAR 10/`) are read and predicted one-by-one.

This simulates **real-world image classification**.

---

## 📈 Performance Tracking

Three curves are plotted using seaborn:

- **Training Loss Curve**
- **Real-Image Test Accuracy Curve (`acc`)**
- **CIFAR-10 Test Accuracy Curve (`test_acc`)**

This helps track model learning progression and generalization.

---

## 🖼️ Real Image Prediction

External images (like real cars, birds, trucks) are:

- Read via OpenCV  
- Preprocessed with the same transforms  
- Forwarded through the trained ResNet  
- Class-name matched to ground-truth filename  

Example:

---

## 💾 Saving the Model

The model weights are saved to:

This file can be used later for deployment, fine-tuning, or conversion to ONNX/TorchScript.

---

## 🛠️ Technologies Used

- **PyTorch**
- **Torchvision**
- **HuggingFace Datasets**
- **OpenCV**
- **NumPy + Pandas**
- **Seaborn**
- **Matplotlib**

---

## 🚀 Future Improvements

- Train deeper networks (ResNet-18, ResNet-34)  
- Add data augmentation (Flip, Crop, Jitter)  
- Switch to full 3-channel (RGB) training  
- Add confusion matrix visualization  
- Integrate Grad-CAM for feature visualization  

---

## 🙌 Acknowledgements

- HuggingFace Datasets team  
- PyTorch community  
- CIFAR-10 dataset creators  

---

If you want, I can also generate:

✅ A cleaner GitHub-styled README  
✅ Badges + model diagram  
✅ Project folder structure  
