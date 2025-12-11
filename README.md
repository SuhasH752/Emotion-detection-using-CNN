Facial Emotion Recognition (FER) on FER2013
This project implements a complete facial emotion recognition pipeline on the FER2013 dataset using multiple CNN architectures, class-imbalance handling, and a Streamlit web application for real-time inference.

📊 Dataset
Dataset: FER2013 (from Kaggle)

Image size: 48×48 pixels, grayscale

Number of classes (7): angry, disgust, fear, happy, neutral, sad, surprise

Split used:

Train: 28,709 images

Test: 7,178 images

Class imbalance is significant, especially for the disgust class, which motivated the use of class weights during training.

🔧 Preprocessing and Augmentation
All images are converted to grayscale and resized.

For the custom CNN (48×48 input):
Grayscale (1 channel)

Resize to 48×48

Random horizontal flip

Random rotation (about 8–15 degrees, depending on experiment)

Mild color jitter (brightness and contrast for robustness)

Normalization (mean = 0.5, std = 0.5)
