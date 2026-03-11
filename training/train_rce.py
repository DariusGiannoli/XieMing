"""
train_rce.py — Train an RCE (Recognition by Components Engine) model.

Uses hand-crafted feature modules (intensity, Sobel, spectral, Laplacian,
gradient orientation, Gabor, LBP) to build a feature vector per image,
then trains a Logistic Regression head on top.
"""

import sys
import os
import cv2
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.detectors.rce.features import REGISTRY
from src.utils import build_rce_vector
from src.config import PROJECT_ROOT as CFG_ROOT, MODEL_PATHS


def load_data():
    """Scan artroom/bird training images and return (images, labels)."""
    images, labels = [], []
    train_dir = CFG_ROOT / "data" / "artroom" / "bird" / "yolo" / "train" / "images"

    print(f"📂 Scanning {train_dir}...")

    for img_file in train_dir.glob("*.png"):
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        filename = img_file.name.lower()

        if "bird" in filename:
            images.append(img)
            labels.append("bird")
        elif any(x in filename for x in ["room", "wall", "floor", "empty"]):
            images.append(img)
            labels.append("background")

    return images, labels


def main():
    print("🚀 Starting RCE Training Pipeline...")
    images, labels = load_data()

    n_bird = labels.count("bird")
    n_bg = labels.count("background")

    print(f"\n📊 Data Summary:")
    print(f"   - Total Images:  {len(images)}")
    print(f"   - Birds:         {n_bird}")
    print(f"   - Backgrounds:   {n_bg}")

    if len(images) == 0:
        print("❌ No images found. Check data/artroom/bird/yolo/train/images/.")
        return

    # All modules active
    active_modules = {k: True for k in REGISTRY}
    active_names = [REGISTRY[k]["label"] for k in active_modules]
    print(f"\n🧬 Active RCE modules: {', '.join(active_names)}")

    # Extract features
    print("\n⏳ Extracting RCE features...")
    X = np.array([build_rce_vector(img, active_modules) for img in images])
    print(f"   Feature matrix shape: {X.shape}")

    # Train Logistic Regression head
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score

    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X, labels)

    train_acc = accuracy_score(labels, model.predict(X))
    print(f"\n📈 Train Accuracy: {train_acc:.2%}")

    if len(images) >= 6:
        cv_scores = cross_val_score(model, X, labels, cv=min(3, len(images) // 2))
        print(f"   CV Accuracy:    {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    # Save model
    save_path = MODEL_PATHS["rce_model"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"\n💾 Model saved to {save_path}")

    # Sanity check
    print("\n🔎 Sanity Check on image 0...")
    vec = build_rce_vector(images[0], active_modules)
    probs = model.predict_proba([vec])[0]
    idx = int(np.argmax(probs))
    print(f"   Result: {model.classes_[idx]} | Confidence: {probs[idx]:.2%}")
    print("\n✅ RCE Training Complete.")


if __name__ == "__main__":
    main()
