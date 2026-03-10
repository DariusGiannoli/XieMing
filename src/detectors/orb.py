import cv2
import numpy as np
import time
import joblib
from pathlib import Path
from src.config import MODEL_PATHS

class ORBDetector:
    """
    The Ancestor: ORB (Oriented FAST and Rotated BRIEF).
    ROBUST VERSION: Tuned for smooth/synthetic data.
    """
    def __init__(self):
        print("🏛️ Initializing Classical ORB Detector (Aggressive Mode)...")
        
        # KEY CHANGE: fastThreshold=0 ensures we detect even faint corners
        # nfeatures=2000 allows us to capture more context
        self.orb = cv2.ORB_create(
            nfeatures=500, 
            scaleFactor=1.2, 
            nlevels=8, 
            edgeThreshold=25, # Reduced from 31 to allow features near edges
            fastThreshold=10  # Reduced from 20 to 0 (Max Sensitivity)
        )
        
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.reference_descriptors = None
        self.model_path = MODEL_PATHS.get('orb_ref')
        self.load_reference()

    def load_reference(self):
        if self.model_path and Path(self.model_path).exists():
            self.reference_descriptors = joblib.load(self.model_path)
            print(f"✅ Loaded ORB Reference from {self.model_path}")
        else:
            print(f"⚠️ Reference not found. Run training/train_orb.py")

    def train(self, images, labels):
        print(f"🏛️ Training ORB on {len(images)} images...")
        
        best_num_features = 0
        best_descriptors = None
        
        bird_images = [img for img, lbl in zip(images, labels) if lbl == 'bird']
        
        if not bird_images:
            print("❌ No bird images found.")
            return

        print(f"   -> Scanning {len(bird_images)} bird images with High Sensitivity...")

        for i, img in enumerate(bird_images):
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # CLAHE: Enhance contrast to help ORB see details in smooth images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            kp, des = self.orb.detectAndCompute(gray, None)
            
            count = 0 if des is None else len(des)
            # print(f"      [Img {i}] Features: {count}") # Uncomment if you need to debug
            
            if des is not None and count > best_num_features:
                best_num_features = count
                best_descriptors = des
        
        if best_descriptors is None:
            raise RuntimeError(
                "ORB training failed: no features detected. "
                "Images may be too smooth, too small, or solid colors.")
        else:
            self.reference_descriptors = best_descriptors
            if self.model_path:
                joblib.dump(self.reference_descriptors, self.model_path)
                print(f"💾 Success! Saved Reference with {best_num_features} features.")

    def predict(self, image):
        if self.reference_descriptors is None:
            return "Untrained", 0.0, 0.0

        t0 = time.perf_counter()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply same contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if des is None:
            return "background", 0.0, (time.perf_counter() - t0) * 1000

        matches = self.bf.match(self.reference_descriptors, des)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Relaxed matching for smooth images
        good_matches = [m for m in matches if m.distance < 70]
        score = len(good_matches)
        
        # Lower threshold for detection
        label = "bird" if score > 5 else "background"
        confidence = min(score / 10.0, 1.0) 
        
        t1 = time.perf_counter()
        return label, confidence, (t1 - t0) * 1000