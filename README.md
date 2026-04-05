# AlexNet-Modern-PyTorch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

A modern implementation of the 8-layer AlexNet architecture using PyTorch, optimized with **Mixed Precision Training (AMP)** for faster performance on the **CIFAR-10** dataset. 



## 🚀 Features
* **8-Layer Architecture:** 5 Convolutional layers and 3 Fully Connected layers, maintaining the classic design.
* **Modern Enhancements:** Optimized for modern GPUs using `torch.cuda.amp` (Mixed Precision).
* **Adaptive Pooling:** Uses `nn.AdaptiveAvgPool2d` to support various input sizes (defaulting to 224x224).
* **High-Speed Loading:** Configured with `num_workers=16` and `pin_memory=True` for high-RAM environments.

---

## 🛠️ Setup and Installation

To get started, clone the repository and ensure you have the necessary libraries installed:

```bash
git clone https://github.com/sankalpalokuliyanage/AlexNet-Modern-PyTorch.git
cd AlexNet-Modern-PyTorch
pip install torch torchvision pillow
```

## 📈 Training
The training script is pre-configured for the CIFAR-10 dataset. It automatically resizes images to 224x224 and applies standard normalization.
```bash
python AlexNet_Training.py
```

Training Specs:
Optimizer: Adam (lr=0.0001)
Batch Size: 512
Epochs: 90
Precision: Mixed Precision (FP16)

## 🔍 Inference (Prediction)
Use the predict.py script to classify a single image. You must provide the image path using the --image flag.
```bash
python alexNet_predict.py
```
or
```bash
py alexNet_predict.py
```

## Run in Google Colab:
Copy and paste these commands into a Colab cell to test the repository:
### 1. Clone the repository
```bash
!git clone https://github.com/sankalpalokuliyanage/AlexNet-Modern-PyTorch.git
%cd AlexNet-Modern-PyTorch
```
### 2. Run inference on a test image (Make sure test_image.jpg exists)
```bash
!python AlexNetColab_predict.py --image test_image.jpg
```


## 📂 Project Structure
direct_alexnet_90epochs.ph - Model trained using 90 epochs | 
AlexNet_Training.py - The training loop with AMP optimization. | 
alexNet_predict.py - Inference script with command-line argument support. | 
README.md - Documentation and usage guide.


## 🎓 Author
L.C. Sankalpa Lokuliyanage Master's Student in Software, Kyungpook National University (KNU), South Korea.
