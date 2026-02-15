import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import joblib
import time
from pathlib import Path
from src.config import MODEL_PATHS

class MobileNetDetector:
    """
    Professional Wrapper for MobileNetV3-Small.
    Target: Ultra-low latency (<3ms) feature extraction for robotics.
    """
    def __init__(self, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"📱 Initializing MobileNetV3 on {self.device}...")
        
        # 1. Initialize Architecture (Small version = Speed)
        self.backbone = models.mobilenet_v3_small(weights=None)
        
        # 2. Load Local Weights (The Backbone)
        model_path = MODEL_PATHS.get('mobilenet')
        if model_path and Path(model_path).exists():
            print(f"📂 Loading backbone from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            try:
                self.backbone.load_state_dict(state_dict)
            except:
                # 'strict=False' is standard when loading backbones for transfer learning
                self.backbone.load_state_dict(state_dict, strict=False)
        else:
            print(f"⚠️ Warning: Local weights not found at {model_path}")

        # 3. Cut off the Classifier
        # We replace the final classifier block with Identity to get raw features
        self.backbone.classifier = torch.nn.Identity()
        
        self.backbone.eval()
        self.backbone.to(self.device)
        
        # 4. Preprocessing (Standard ImageNet stats)
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 5. Load the Head (The Brain we train)
        # We auto-generate the head path based on the model path
        self.head_path = str(model_path).replace('.pth', '_head.pkl')
        self.head = None
        self.load_head()

    def load_head(self):
        if Path(self.head_path).exists():
            self.head = joblib.load(self.head_path)
            print(f"✅ Loaded trained head from {self.head_path}")
        else:
            print(f"⚠️ Head not found. Model is in FEATURE ONLY mode.")

    def _get_features(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.backbone(input_tensor)
        
        return features.cpu().numpy().flatten()

    def train_head(self, images, labels):
        from sklearn.linear_model import LogisticRegression
        
        if not images:
            raise ValueError("No images provided.")

        print(f"⏳ Extracting features for {len(images)} images...")
        X_data = [self._get_features(img) for img in images]
        
        print("🎓 Fitting Logistic Regression...")
        self.head = LogisticRegression(max_iter=1000)
        self.head.fit(X_data, labels)
        
        joblib.dump(self.head, self.head_path)
        print(f"💾 Model saved to {self.head_path}")

    def predict(self, image):
        if self.head is None:
            return "Untrained", 0.0, 0.0

        t0 = time.perf_counter()
        
        features = self._get_features(image)
        probs = self.head.predict_proba([features])[0]
        winner_idx = np.argmax(probs)
        
        label = self.head.classes_[winner_idx]
        conf = probs[winner_idx]
        
        t1 = time.perf_counter()
        return label, conf, (t1 - t0) * 1000