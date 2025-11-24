# **Emotion Detection from Images**

A deep learning web application that detects human emotions from uploaded images or camera captures using Convolutional Neural Networks.

---

## **Project Overview**

This system classifies facial expressions into 7 emotion categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The application uses multiple CNN architectures trained on the FER2013 dataset.

---

## **Features**

- Upload images or capture directly from camera
- Real-time emotion detection with confidence scores
- Multiple model options with different accuracy levels
- Automatic face detection and cropping
- Adjustable confidence threshold

---

## **Installation**

### **Prerequisites**
- Python 3.8+
- PyTorch
- OpenCV
- Streamlit

### **Setup**

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
```

2. **Create virtual environment**
```bash
python -m venv emotion_env
source emotion_env/bin/activate  # On Windows: emotion_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download trained models and place in `models/` directory**

---

## **Usage**

### **Running the Application**

1. **Start the Streamlit server:**
```bash
streamlit run app.py
```

2. **Open your browser to `http://localhost:8501`**

3. **Use the application:**
   - Select model type from available options
   - Adjust confidence threshold
   - Choose input method (upload or camera)
   - Click "Detect Emotions"

---

## **Model Performance**

| Model Type | Architecture | Accuracy | Key Features |
|------------|--------------|----------|--------------|
| Custom CNN | Custom 3-layer CNN | 53.78% | Basic architecture, batch normalization |
| VGG16 | VGG16 with transfer learning | 57.06% | Pretrained weights, fine-tuned classifier |
| EfficientNet-B0 | EfficientNet-B0 | 46.78% | Lightweight architecture, faster inference |
| ResNet34 | ResNet34 with transfer learning | 52.83% | Residual connections, moderate complexity |
| ResNet18 | ResNet18 with transfer learning | 56.48% | Balanced performance and speed |
| Optimized ResNet18 | Enhanced ResNet18 | 61.99% | Improved classifier, better regularization |
| Ensemble ResNet18 | 3 ResNet18 models | 63.46% | Model averaging, reduced variance |
| Ensemble + TTA | Ensemble with test-time augmentation | 63.78% | Horizontal flip augmentation, most robust |

---

## **Best Performing Models**

**Primary Models Available in App:**
- **Ensemble ResNet18 + TTA**: 63.78% (Recommended)
- **Optimized ResNet18**: 61.99% (Faster inference)

---

## **Training Details**

- **Dataset**: FER2013 (35,887 images)
- **Classes**: 7 emotions with significant class imbalance
- **Class Handling**: Weighted loss functions and data sampling
- **Augmentation**: Random flips, rotations, affine transformations
- **Validation**: 20% holdout set from FER2013 test split

---

## **Model Architecture Details**

### **Ensemble ResNet18 (Best Performance)**
- **Base**: 3 ResNet18 models with different initializations
- **Input**: 48x48 grayscale converted to 3 channels
- **Classifier**: Custom layers with dropout (0.5) and batch normalization
- **Training**: 35 epochs with weighted cross-entropy loss
- **Inference**: Test-time augmentation with horizontal flips

### **Optimized ResNet18**
- **Base**: ResNet18 with frozen early layers
- **Classifier**: 256-128-7 architecture with dropout
- **Optimizer**: Adam with learning rate 0.0001
- **Regularization**: Weight decay and extensive dropout

---

## **Project Structure**

```
emotion-detection/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── models/               # Trained model weights
│   ├── best_ensemble_model.pth
│   └── best_single_model.pth
├── training/             # Training scripts and notebooks
│   ├── model_training.ipynb
│   └── data_preprocessing.py
└── README.md
```

---

## **Dependencies**

```
streamlit==1.28.0
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.0.74
numpy==1.24.3
Pillow==10.0.0
```

---

## **Technical Details**

- **Face Detection**: Haar Cascade Classifier (OpenCV)
- **Image Preprocessing**: 48x48 resize, normalization, format conversion
- **Input Support**: JPEG, PNG, BMP formats; grayscale and color images
- **Camera Integration**: Real-time capture via device camera

---

## **Deployment**

### **Local**
```bash
streamlit run app.py
```
## **Acknowledgments**

- FER2013 dataset providers
- PyTorch and Streamlit communities
- OpenCV for face detection capabilities

---
