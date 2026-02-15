import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import joblib
import time
from pathlib import Path
from src.config import MODEL_PATHS

class ResNetDetector:
    """
    Wrapper for ResNet-18 Feature Extractor.
    Architecture: Local Frozen ResNet-18 Backbone + Logistic Regression Head.
    """
    def __init__(self, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        # 1. Initialize the Architecture (Empty)
        self.backbone = models.resnet18(weights=None)
        
        # 2. Load YOUR Local Weights
        resnet_path = MODEL_PATHS['resnet'] # Defined in config.py
        if Path(resnet_path).exists():
            print(f"Loading local weights from {resnet_path}")
            state_dict = torch.load(resnet_path, map_location=self.device)
            
            try:
                self.backbone.load_state_dict(state_dict)
            except RuntimeError as e:
                print("loading failed:", e)
                self.backbone.load_state_dict(state_dict, strict=False)
        else:
            print("Error Loading")

        # 3. Prepare for Feature Extraction
        self.backbone.eval() # Freeze layers
        self.backbone.to(self.device)
        
        # Remove the final classification layer
        self.feature_extractor = torch.nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 4. Define Preprocessing (Standard ImageNet stats)
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 5. Load the Head (trained brain)
        self.head_path = MODEL_PATHS.get('resnet_head')
        self.head = None
        self.load_head()

    def load_head(self):
        """Loads the trained Logistic Regression head if it exists."""
        if self.head_path and Path(self.head_path).exists():
            self.head = joblib.load(self.head_path)
            print(f"Loaded trained head from {self.head_path}")
        else:
            print(f"No trained head found at {self.head_path}")

    def _get_features(self, img):
        """Internal method to turn an image into a math vector."""
        # Convert BGR (OpenCV) to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess to tensor
        input_tensor = self.preprocess(img_rgb).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
        
        # Flatten [1, 512, 1, 1] -> [512] for Scikit-Learn
        return features.cpu().numpy().flatten()

    def train_head(self, images, labels):
        """
        Trains the lightweight decision layer on top of your local ResNet.
        """
        from sklearn.linear_model import LogisticRegression
        
        if not images:
            raise ValueError("No images provided for training.")

        print(f"⏳ Extracting features for {len(images)} images...")
        X_data = [self._get_features(img) for img in images]
        
        print("🎓 Fitting Logistic Regression...")
        self.head = LogisticRegression(max_iter=1000, C=1.0)
        self.head.fit(X_data, labels)
        
        # Save immediately
        if self.head_path:
            joblib.dump(self.head, self.head_path)
            print(f"💾 Model saved to {self.head_path}")

    def predict(self, image):
        """
        Standard Interface: Returns (Label, Confidence, Time_ms)
        """
        if self.head is None:
            return "Untrained", 0.0, 0.0

        t0 = time.perf_counter()
        
        # 1. Get Vector
        features = self._get_features(image)
        
        # 2. Get Probabilities
        probs = self.head.predict_proba([features])[0]
        winner_idx = np.argmax(probs)
        
        label = self.head.classes_[winner_idx]
        conf = probs[winner_idx]
        
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000
        
        return label, conf, inference_ms