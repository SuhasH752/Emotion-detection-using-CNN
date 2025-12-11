import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, mobilenet_v3_small, resnet18
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="FER App", page_icon=":)", layout="wide")

MODELS_PATH = Path(r"C:\Users\Admin\emod\models")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class MobileNetFER(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        base_model = mobilenet_v3_small(weights=None)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            base_model.classifier[0],
            base_model.classifier[1],
            nn.Linear(base_model.classifier[3].in_features, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class EfficientNetFER(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        base_model = efficientnet_b0(weights=None)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(base_model.classifier[1].in_features, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNetFER(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        base_model = resnet18(weights=None)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(base_model.fc.in_features, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    eff_model = EfficientNetFER(num_classes=7).to(device)
    eff_checkpoint = torch.load(MODELS_PATH / 'best_efficientnet_b0.pth', map_location=device)
    eff_model.load_state_dict(eff_checkpoint['model_state_dict'])
    eff_model.eval()
    
    mob_model = MobileNetFER(num_classes=7).to(device)
    mob_checkpoint = torch.load(MODELS_PATH / 'best_mobilenetv3_small.pth', map_location=device)
    mob_model.load_state_dict(mob_checkpoint['model_state_dict'])
    mob_model.eval()
    
    res_model = ResNetFER(num_classes=7).to(device)
    res_checkpoint = torch.load(MODELS_PATH / 'best_resnet18.pth', map_location=device)
    res_model.load_state_dict(res_checkpoint['model_state_dict'])
    res_model.eval()
    
    return {
        'efficientnet': eff_model, 
        'mobilenet': mob_model, 
        'resnet': res_model
    }, device

models, device = load_models()

eff_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mob_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

resnet_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detect_face(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_img)
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        h, w, _ = img.shape
        x, y = int(bbox.xmin * w), int(bbox.ymin * h)
        w_face, h_face = int(bbox.width * w), int(bbox.height * h)
        face = img[y:y+h_face, x:x+w_face]
        return cv2.resize(face, (224, 224))
    return cv2.resize(img, (224, 224))

def predict_emotion(image, model_name='efficientnet'):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_face = detect_face(img_cv)
    pil_img = Image.fromarray(cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB))
    
    if model_name == 'efficientnet':
        input_tensor = eff_transform(pil_img).unsqueeze(0).to(device)
        model = models['efficientnet']
    elif model_name == 'mobilenet':
        input_tensor = mob_transform(pil_img).unsqueeze(0).to(device)
        model = models['mobilenet']
    else:
        input_tensor = resnet_transform(pil_img).unsqueeze(0).to(device)
        model = models['resnet']
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_id = torch.argmax(probs).item()
        confidence = probs[pred_id].item()
    
    return class_names[pred_id], confidence, probs.cpu().numpy()

st.title("Facial Emotion Recognition App")
st.markdown("71.3% EfficientNet-B0 | 69.8% MobileNetV3 | 69.3% ResNet18 | Real-time Detection")

st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Select Model:",
    ['EfficientNet-B0 (71.3%)', 'MobileNetV3-Small (69.8%)', 'ResNet18 (69.3%)']
)

if 'EfficientNet' in model_choice:
    model_name = 'efficientnet'
elif 'MobileNet' in model_choice:
    model_name = 'mobilenet'
else:
    model_name = 'resnet'

st.sidebar.markdown("""
### Leaderboard
| Model            | Test Acc |
|------------------|----------|
| EfficientNet-B0  | 71.3%    |
| MobileNetV3      | 69.8%    |
| ResNet18         | 69.3%    |
| CustomCNN        | 64.4%    |
""")

tab1, tab2 = st.tabs(["Upload Image", "Live Webcam"])

with tab1:
    uploaded_file = st.file_uploader("Choose a face image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Input Image", use_column_width=True)
        
        with col2:
            emotion, confidence, probs = predict_emotion(image, model_name)
            st.markdown(f"### Predicted: {emotion.upper()}")
            st.metric("Confidence", f"{confidence:.1%}")
            st.progress(confidence)
            
            top3_idx = np.argsort(probs)[-3:][::-1]
            with st.expander("Top 3 Emotions"):
                for i, idx in enumerate(top3_idx):
                    st.write(f"{i+1}. {class_names[idx]}: {probs[idx]:.1%}")

with tab2:
    st.header("Live Detection")
    camera_img = st.camera_input("Take a selfie")
    
    if camera_img:
        image = Image.open(camera_img).convert('RGB')
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Live Capture", use_column_width=True)
        
        with col2:
            emotion, confidence, probs = predict_emotion(image, model_name)
            st.markdown(f"### Live Emotion: {emotion.upper()}")
            st.metric("Confidence", f"{confidence:.1%}")
            
            gauge_value = int(confidence * 100)
            st.markdown(
                f"""
                <div style="text-align: center; padding: 20px;">
                    <h2 style="color: #1f77b4;">{gauge_value}%</h2>
                    <div style="background: linear-gradient(90deg, #ff4444 0%, #44ff44 {gauge_value}%);
                               height: 25px; border-radius: 15px; border: 3px solid #ddd;">
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

st.markdown("---")
st.markdown("Production-ready FER app with trained models (71.3% top accuracy)")
