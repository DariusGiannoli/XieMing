import torch
import timm
import cv2
import numpy as np
import joblib
import time
from pathlib import Path
from src.config import MODEL_PATHS

class MobileViTDetector:
    """
    The Modern Challenger: MobileViT-XXS.
    Architecture: Hybrid CNN + Transformer.
    Source: Apple Research (2022).
    """
    def __init__(self, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🤖 Initializing MobileViT-XXS on {self.device}...")
        
        # 1. Load the Backbone (Pretrained on ImageNet)
        # 'num_classes=0' cuts off the classification head so we get raw features
        # We load from local file if available, otherwise timm downloads it
        try:
            self.backbone = timm.create_model('mobilevit_xxs.cvnets_in1k', pretrained=True, num_classes=0)
        except Exception as e:
            print(f"⚠️ Could not download model: {e}")
            print("   -> Ensure you have internet or run 'scripts/download_models.py' first.")
            return

        self.backbone.eval()
        self.backbone.to(self.device)
        
        # 2. Auto-Setup Preprocessing
        # Transformers are picky about size (usually 256x256). We ask the model what it wants.
        config = timm.data.resolve_model_data_config(self.backbone)
        self.transforms = timm.data.create_transform(**config, is_training=False)
        
        # 3. Load the Head (The Brain we train)
        self.head_path = MODEL_PATHS.get('mobilevit_head')
        self.head = None
        self.load_head()

    def load_head(self):
        if self.head_path and Path(self.head_path).exists():
            self.head = joblib.load(self.head_path)
            print(f"✅ Loaded MobileViT head from {self.head_path}")
        else:
            print(f"⚠️ Head not found. Run training/train_mobilevit.py")

    def _get_features(self, img):
        """
        Converts image to feature vector using the Hybrid Backbone.
        """
        # Convert to PIL (Required for timm transforms)
        from PIL import Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Transform and Add Batch Dimension [1, C, H, W]
        input_tensor = self.transforms(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.backbone(input_tensor)
        
        # Flatten [1, 320] -> [320]
        return features.cpu().numpy().flatten()

    def train_head(self, images, labels):
        from sklearn.linear_model import LogisticRegression
        
        if not images:
            raise ValueError("No images provided.")

        print(f"⏳ Extracting features for {len(images)} images (Slower than CNNs)...")
        X_data = [self._get_features(img) for img in images]
        
        print("🎓 Fitting Logistic Regression...")
        self.head = LogisticRegression(max_iter=1000)
        self.head.fit(X_data, labels)
        
        if self.head_path:
            joblib.dump(self.head, self.head_path)
            print(f"💾 Model saved to {self.head_path}")

    def predict(self, image):
        if self.head is None: return "Untrained", 0.0, 0.0
        
        t0 = time.perf_counter()
        
        # 1. Feature Extraction
        feats = self._get_features(image)
        
        # 2. Prediction
        probs = self.head.predict_proba([feats])[0]
        idx = np.argmax(probs)
        
        label = self.head.classes_[idx]
        conf = probs[idx]
        
        t1 = time.perf_counter()
        return label, conf, (t1 - t0) * 1000

    def get_activation_maps(self, img, n_maps: int = 6):
        """
        Returns n_maps normalised float32 arrays from the last transformer stage.
        Each array is a single channel spatial activation map in [0, 1].
        """
        from PIL import Image
        captured = {}
        hook = self.backbone.stages[-1].register_forward_hook(
            lambda m, i, o: captured.update({"feat": o})
        )
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = self.transforms(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.backbone(tensor)
        hook.remove()
        acts = captured["feat"][0].cpu().numpy()  # (C, H, W)
        maps = []
        for i in range(min(n_maps, acts.shape[0])):
            m = acts[i]
            m = (m - m.min()) / (m.max() - m.min() + 1e-5)
            maps.append(m.astype(np.float32))
        return maps