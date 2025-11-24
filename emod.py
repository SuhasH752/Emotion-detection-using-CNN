import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import cv2
import os

st.set_page_config(page_title="Emotion Detection", layout="centered")

class OptimizedEmotionResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super(OptimizedEmotionResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

class EnsembleModel(nn.Module):
    def __init__(self, num_models=3, num_classes=7):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList()
        for _ in range(num_models):
            self.models.append(OptimizedEmotionResNet18(num_classes))
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(0)

def load_emotion_model(model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)
    
    if checkpoint.get('model_type') == 'Ensemble_ResNet18':
        model = EnsembleModel()
    else:
        model = OptimizedEmotionResNet18(num_classes=checkpoint['num_classes'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['class_names']

def detect_faces_robust(image):
    image_np = np.array(image)
    
    if len(image_np.shape) == 2:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    else:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=1, minSize=(10, 10))
    
    return faces, image_cv

def preprocess_face(face_roi, img_size=48):
    if len(face_roi.shape) == 2:
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
    elif face_roi.shape[2] == 3:
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    else:
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGRA2RGB)
    
    face_pil = Image.fromarray(face_rgb)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transform(face_pil).unsqueeze(0)

def predict_emotion(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        all_confidences = probabilities.squeeze().tolist()
    
    return class_names[predicted.item()], confidence.item(), all_confidences

def draw_face_boxes(image_cv, faces, emotions, confidences):
    result_image = image_cv.copy()
    
    for (x, y, w, h), emotion, confidence in zip(faces, emotions, confidences):
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{emotion} ({confidence:.1%})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result_image, (x, y-25), (x+label_size[0], y), (0, 255, 0), -1)
        cv2.putText(result_image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

def process_image_with_model(image, model, class_names, confidence_threshold):
    faces, image_cv = detect_faces_robust(image)
    
    if len(faces) == 0:
        return None, None, "No faces detected"
    
    emotions = []
    confidences = []
    all_predictions = []
    
    for i, (x, y, w, h) in enumerate(faces):
        face_roi = image_cv[y:y+h, x:x+w]
        face_tensor = preprocess_face(face_roi)
        emotion, confidence, all_probs = predict_emotion(model, face_tensor, class_names)
        
        if confidence >= confidence_threshold:
            emotions.append(emotion)
            confidences.append(confidence)
            all_predictions.append((emotion, confidence, all_probs))
        else:
            emotions.append("Low Confidence")
            confidences.append(confidence)
            all_predictions.append(("Low Confidence", confidence, all_probs))
    
    result_image = draw_face_boxes(image_cv, faces, emotions, confidences)
    return result_image, all_predictions, f"Detected {len(faces)} face(s)"

def main():
    st.title("Emotion Detection from Images")
    
    st.sidebar.title("Settings")
    model_choice = st.sidebar.selectbox(
        "Choose Model:",
        ["Ensemble Model (63.78% Accuracy)", "Single Model (61.99% Accuracy)"]
    )
    
    confidence_threshold = st.sidebar.slider(
        "Minimum Confidence Threshold:",
        min_value=0.1, max_value=0.9, value=0.3
    )
    
    input_method = st.radio("Choose Input Method:", ["Upload Image", "Camera Capture"])
    
    if "Ensemble" in model_choice:
        model_path = "models/best_ensemble_model.pth"
    else:
        model_path = "models/best_single_model.pth"
    
    model, class_names = load_emotion_model(model_path)
    
    if model is None:
        st.error("Failed to load emotion detection model")
        return
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'bmp'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            if st.button("Detect Emotions", type="primary"):
                with st.spinner("Processing image..."):
                    result_image, predictions, message = process_image_with_model(
                        image, model, class_names, confidence_threshold
                    )
                    
                    if result_image is not None:
                        # Reduced image size - width of 400 pixels
                        st.image(result_image, caption=message, width=400)
                        
                        if predictions:
                            for i, (emotion, confidence, all_probs) in enumerate(predictions):
                                st.subheader(f"Face {i+1}: {emotion} ({confidence:.1%} confidence)")
                                
                                for cls_name, prob in zip(class_names, all_probs):
                                    col_a, col_b, col_c = st.columns([2, 5, 1])
                                    col_a.write(cls_name)
                                    col_b.progress(float(prob))
                                    col_c.write(f"{prob:.1%}")
                    else:
                        st.warning(message)
    
    else:  # Camera Capture
        camera_image = st.camera_input("Take a picture with your camera")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            
            if st.button("Detect Emotions", type="primary"):
                with st.spinner("Processing image..."):
                    result_image, predictions, message = process_image_with_model(
                        image, model, class_names, confidence_threshold
                    )
                    
                    if result_image is not None:
                        # Reduced image size - width of 400 pixels
                        st.image(result_image, caption=message, width=400)
                        
                        if predictions:
                            for i, (emotion, confidence, all_probs) in enumerate(predictions):
                                st.subheader(f"Face {i+1}: {emotion} ({confidence:.1%} confidence)")
                                
                                for cls_name, prob in zip(class_names, all_probs):
                                    col_a, col_b, col_c = st.columns([2, 5, 1])
                                    col_a.write(cls_name)
                                    col_b.progress(float(prob))
                                    col_c.write(f"{prob:.1%}")
                    else:
                        st.warning(message)

if __name__ == "__main__":
    main()