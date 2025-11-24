🎯 Project Overview
This project implements a real-time emotion detection system that can classify facial expressions into 7 emotion categories:

😠 Angry

🤢 Disgust

😨 Fear

😊 Happy

😐 Neutral

😢 Sad

😲 Surprise

The system uses ensemble learning with ResNet18 architecture and achieves 63.78% accuracy on the FER2013 dataset.

🚀 Features
Dual Input Methods: Upload images or capture directly from camera

Real-time Processing: Instant emotion detection with confidence scores

Multiple Models: Choose between Ensemble (63.78%) or Single (61.99%) model

Face Detection: Automatic face detection and cropping

Confidence Thresholding: Adjustable confidence levels for predictions

Responsive UI: Clean Streamlit-based web interface

🛠️ Installation
Prerequisites
Python 3.8+

PyTorch

OpenCV

Streamlit

Step-by-Step Setup
Clone the repository

bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
Create virtual environment (Recommended)

bash
python -m venv emotion_env
source emotion_env/bin/activate  # On Windows: emotion_env\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Download trained models

Place best_ensemble_model.pth and best_single_model.pth in the models/ directory

Download models from releases

📁 Project Structure
text
emotion-detection/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── models/               # Trained model weights
│   ├── best_ensemble_model.pth
│   └── best_single_model.pth
├── training/             # Model training scripts
│   ├── train_models.py
│   └── data_loader.py
└── README.md
🎮 Usage
Running the Application
Start the Streamlit server

bash
streamlit run app.py
Access the web interface

Open your browser and go to http://localhost:8501

Using the application

Select Model: Choose between Ensemble or Single model

Adjust Confidence: Set minimum confidence threshold (0.1-0.9)

Choose Input: Upload image or use camera capture

Detect Emotions: Click "Detect Emotions" button

Example Usage
python
# For custom integration
from app import load_emotion_model, process_image_with_model

model, class_names = load_emotion_model('models/best_ensemble_model.pth')
result_image, predictions, message = process_image_with_model(
    image, model, class_names, confidence_threshold=0.5
)
🧠 Model Architecture
Ensemble ResNet18
Base Model: ResNet18 with modified first layer for grayscale input

Classifier: Custom fully-connected layers with dropout and batch normalization

Ensemble: 3 independently trained models with averaged predictions

Input Size: 48×48 grayscale images

Output: 7 emotion classes with confidence scores

Training Details
Dataset: FER2013 (35,887 images)

Accuracy: 63.78% (Ensemble), 61.99% (Single)

Class Handling: Weighted loss for imbalanced data

Augmentation: Random flips, rotations, and affine transformations

📊 Performance
Model Type	Accuracy	Precision	Recall	F1-Score
Single Model	61.99%	0.62	0.61	0.61
Ensemble Model	63.78%	0.64	0.63	0.63
Class-wise Performance (Ensemble Model):

Happy: 85% accuracy

Surprise: 72% accuracy

Neutral: 65% accuracy

Angry: 58% accuracy

Sad: 55% accuracy

Fear: 48% accuracy

Disgust: 45% accuracy

🔧 Technical Details
Dependencies
txt
streamlit==1.28.0
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.0.74
numpy==1.24.3
Pillow==10.0.0
Face Detection
Method: Haar Cascade Classifier (OpenCV)

Features: Multi-scale detection with fallback mechanisms

Robustness: Handles various image formats and lighting conditions

Preprocessing Pipeline
Face detection and cropping

Resize to 48×48 pixels

Grayscale to 3-channel conversion

Normalization (mean=0.5, std=0.5)

Batch processing for model inference

🎯 API Reference
Main Functions
load_emotion_model(model_path, device='cpu')
Loads trained emotion detection model.

Parameters:

model_path: Path to model weights file

device: Processing device ('cpu' or 'cuda')

Returns:

model: Loaded PyTorch model

class_names: List of emotion classes

process_image_with_model(image, model, class_names, confidence_threshold)
Processes image and returns emotion predictions.

Parameters:

image: PIL Image object

model: Loaded emotion model

class_names: Emotion class names

confidence_threshold: Minimum confidence score (0.1-0.9)

Returns:

result_image: Image with bounding boxes

predictions: List of emotion predictions

message: Processing status message

🚀 Deployment
Local Deployment
bash
streamlit run app.py
