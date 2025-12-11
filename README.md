
# Facial Emotion Recognition 

This project implements a complete facial emotion recognition pipeline using multiple CNN architectures trained on the FER2013 dataset. It includes preprocessing, class-imbalance handling, training scripts, evaluation metrics, and a Streamlit web application for real-time inference.

---

## Dataset: FER2013

* Source: Kaggle
* Image size: 48×48 pixels, grayscale
* Number of classes: 7

  * angry
  * disgust
  * fear
  * happy
  * neutral
  * sad
  * surprise
* Train samples: 28,709
* Test samples: 7,178
* Significant class imbalance, especially for the disgust class.

---

## Preprocessing and Data Augmentation

### Custom CNN (48×48 grayscale)

* Convert to grayscale
* Resize to 48×48
* Random horizontal flip
* Random rotation (8–15 degrees)
* Mild brightness and contrast jitter
* Normalize (mean = 0.5, std = 0.5)

### Pretrained Models (3-channel, 224–240 px)

* Convert grayscale to RGB (3 channels)
* Resize to 224×224 or 240×240
* Random horizontal flip
* Random rotation
* Mild color jitter
* Normalize with ImageNet statistics

---

## Class Imbalance Handling

Class weights computed using:

wᵢ = total_samples / (num_classes × class_countᵢ)

Approximate class weights used:

```
[1.03, 9.41, 1.00, 0.57, 0.83, 0.85, 1.29]
```

This improves performance for minority classes such as disgust.

---

## Trained Models

### 1. Custom CNN (from scratch)

* Three convolutional blocks
* Each block: Conv → BatchNorm → ReLU (twice)
* MaxPool after each block
* Global average pooling
* Fully connected layers with dropout and batch normalization
* Optimizer: AdamW
* Loss: CrossEntropyLoss with class weights
* Epochs: 50–75

Results:

* Test accuracy: 64.2–64.4%
* Train accuracy: ~67–68%
* Small generalization gap

---

### 2. MobileNetV3-Small (pretrained)

* Pretrained on ImageNet
* Classifier head replaced with 7-class output
* Early layers partially frozen
* Input: 3×224×224
* Optimizer: AdamW
* Scheduler: ReduceLROnPlateau

Results:

* Test accuracy: 69.8%
* Train accuracy: ~98%
* Per-class performance:

  * happy, surprise: strong
  * disgust F1: ~0.72
  * fear, sad: moderate but improved

---

### 3. EfficientNet-B0 (pretrained) – Best Model

* Pretrained on ImageNet
* Custom classifier with dropout
* Early blocks frozen initially, then fine-tuned
* Input: 3×240×240

Results:

* Best test accuracy: 71.3%
* Final test accuracy: 70.9%
* Train accuracy: ~98%
* Macro F1: ~0.70
* Per-class F1:

  * happy: ~0.89
  * surprise: ~0.83
  * disgust: ~0.75
  * others: 0.56–0.66

This model is used as the default in the application.

---

### 4. ResNet18 (pretrained)

* Pretrained on ImageNet
* Final fully connected layer replaced
* Input: 3×224×224

Results:

* Test accuracy: 69.3%
* Train accuracy: 98–99%
* Very stable validation performance

---

## Summary of Results

| Model           | Type       | Test Accuracy | Notes                       |
| --------------- | ---------- | ------------- | --------------------------- |
| Custom CNN      | Scratch    | ~64.4%        | Strong 48×48 baseline       |
| MobileNetV3     | Pretrained | 69.8%         | Lightweight, fast inference |
| EfficientNet-B0 | Pretrained | 71.3%         | Best-performing model       |
| ResNet18        | Pretrained | 69.3%         | Stable and reliable         |

All models use the same train/test split and class weighting strategy.

---

## Training Artifacts

Each model saves the following in the `models/` directory:

1. best_<model_name>.pth

   * model_state_dict
   * best epoch
   * best train and test accuracy
   * class weights

2. <model_name>_metrics.pth

   * training history
   * validation metrics

3. <model_name>_training_curves.png

   * accuracy vs epoch

4. <model_name>_confusion_matrix.png

   * test-set confusion matrix

---

## Streamlit Application

A Streamlit app (`app.py`) is provided for real-time FER.

Features:

* Select model: EfficientNet-B0, MobileNetV3, ResNet18
* Image upload
* Webcam capture
* Automatic face detection using MediaPipe
* Outputs:

  * Top-1 emotion
  * Confidence score
  * Top-3 probabilities

Run the app:

```
streamlit run app.py
```

