# Breast Cancer Histopathology Classifier (400X)

This project uses **transfer learning** with a pre-trained ResNet18 model to classify breast cancer **histopathology images (400X magnification)** into **benign** or **malignant** categories.

---

## 📊 Overview

- **Model**: ResNet18 (pretrained on ImageNet)
- **Dataset**: [BreaKHis 400X subset](https://www.kaggle.com/datasets/ambarish/breakhis) — H&E stained breast tumor images
- **Task**: Binary classification (benign vs. malignant)
- **Accuracy Achieved**: ~95%
- **Libraries Used**: PyTorch, torchvision, matplotlib, scikit-learn

---

## 🗂️ Dataset Structure

The dataset should be organized as follows:
breakhis_data/
└── BreaKHis 400X/
├── train/
│ ├── benign/
│ └── malignant/
└── test/
├── benign/
└── malignant/


Each folder contains PNG slide images at 400X magnification.

---

## 🧠 Model Details

We use transfer learning on **ResNet18**, replacing the final fully connected layer with:

```python
nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)

