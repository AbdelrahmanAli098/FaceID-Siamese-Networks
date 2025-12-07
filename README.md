# Face Verification & Recognition System (Siamese Neural Network + Kivy GUI)

A real-time face verification system built using a **Siamese neural network** and a **Kivy-based desktop application**.  
This project performs **live face capture**, processes the image using a trained deep learning model, and verifies the identity by comparing it with reference images.

---

## Features

- **Deep Learning Siamese Model** for face similarity measurement  
- **Real-time webcam feed** using OpenCV  
- **Kivy-based graphical user interface**  
- Fully automated preprocessing: resizing, normalization, encoding  
- Verification using multiple stored reference images  
- Custom **L1 Distance Layer** used for comparing embeddings  
- Threshold-based decision logic for robust verification  
- Fast inference suitable for real-time use  

---

## How the System Works

### 1. **Model (Siamese Network – Trained in the Notebook)**  
- Uses a dual-branch convolutional network to encode two face images  
- Computes similarity using a custom-built **L1 Distance layer**  
- Outputs a probability between **0 (different person)** and **1 (same person)**  
- Trained on image pairs (positive & negative samples)

### 2. **GUI Application (FaceID.py)**  
- Loads the trained `siamese_modelv2.keras` model  
- Displays a live webcam stream  
- On button click:
  - Captures the current face image  
  - Preprocesses it to match model requirements  
  - Compares it with all images in `Test/VerificationImages/`  
  - Aggregates the results and decides **Verified** / **Unverified**

---

## Project Structure

```plaintext
FaceVerification/
│
├─ app/
│   ├─ FaceID.py
│   ├─ layers.py
│   ├─ siamese_modelv2.keras          # Trained Siamese model
│   └─ Test/
│       ├─ InputImages/
│       │   └─ InputImages.jpg        # Automatically saved face capture
│       └─ VerificationImages/        # Reference face images
├─ Data/
│   ├─ Anchors/                       # Face anchor images
│   ├─ Positive/                      # Positive face images
│   └─ Negative/                      # LFW dataset images
│
├─ Test/                              # Another test set (same structure)
│   ├─ InputImages/
│   └─ VerificationImages/
│
├─ training_checkpoints/              # Saved checkpoints during model training
├─ FaceVerification.ipynb             # Jupyter notebook used for model training
├─ Siamese Neural Networks for One-shot Image Recognition.pdf
└─ siamese_modelv2.keras              # Final trained Siamese model
```
---

## 🚀 Features

- Real-time webcam feed inside the Kivy UI  
- Automatic face capture & preprocessing  
- Siamese neural network for similarity scoring  
- Adjustable detection & verification thresholds  
- Clean and simple GUI with live feedback  

---
