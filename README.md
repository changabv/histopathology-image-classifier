# Histopathology Image Classifier (BreaKHis 400X)

This project uses transfer learning to classify H&E-stained breast tissue images as **benign** or **malignant**. It’s built with PyTorch and uses the [BreaKHis 400X dataset](https://www.kaggle.com/datasets/forderation/breakhis-400x).

## Overview

- **Model**: ResNet18 with a custom classification head  
- **Task**: Binary image classification (benign vs malignant)  
- **Dataset**: BreaKHis 400X (breast cancer histopathology)  
- **Tools**: PyTorch, torchvision, sklearn, matplotlib  

## Dataset

- Source: [Kaggle - BreaKHis 400X](https://www.kaggle.com/datasets/forderation/breakhis-400x)  
- Format: Pre-split into `train/` and `test/` folders by class  
- Total images used: ~1,700  

**Classes**:
- `benign`
- `malignant`

## Model Architecture

We use a pretrained ResNet18 from `torchvision.models`, replacing the final fully-connected layer with:

```python
nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)
```

## Training Details

- **Epochs**: 5 (can be increased)  
- **Optimizer**: Adam (lr=1e-4)  
- **Loss Function**: CrossEntropyLoss  
- **Hardware**: CPU / GPU-compatible  

## Performance

Achieved ~95% accuracy on the test set.

```
              precision    recall  f1-score   support

      benign       0.90      0.95      0.93       176
   malignant       0.98      0.95      0.96       369

    accuracy                           0.95       545
```

## Visualization

- Confusion matrix and classification report  
- Random visual samples of correct and incorrect predictions  
- Grad-CAM to visualize model attention

<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/8d71e510-ce97-4635-b62c-54e47719dd7b" />

 <img width="1317" height="985" alt="image" src="https://github.com/user-attachments/assets/2ff7381c-8bd3-499d-baea-c373d47cff90" />

<img width="990" height="355" alt="image" src="https://github.com/user-attachments/assets/29959fcb-be80-4f13-a9cd-384a0b4366ec" />


## How to Run

1. Clone the repo and upload the `BreaKHis 400X` dataset.  
2. Place the dataset in `breakhis_data/BreaKHis 400X/`  
3. Run the training script (e.g., in Google Colab or Jupyter).  
4. Outputs:
    - Performance metrics
    - Confusion matrix
    - Sample predictions
    - Grad-CAM visualizations

## Citation

> Forderation. *BreaKHis 400X*. Kaggle, 2021. https://www.kaggle.com/datasets/forderation/breakhis-400x

## Author

Built by Yudam Chang as a portfolio project in cancer image classification using deep learning.
