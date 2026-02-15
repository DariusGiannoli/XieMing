import sys
import os
import cv2
from pathlib import Path

# Add project root to path so we can import 'src'
# We use .parent because this script is inside 'training/'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.detectors.resnet import ResNetDetector
from src.config import PROJECT_ROOT

def load_data():
    """
    Scans the data folders and prepares clean lists for training.
    """
    images = []
    labels = []
    
    # 1. Load BIRDS (Positive)
    # Using the path from your tree structure
    train_dir = PROJECT_ROOT / "data/artroom/bird/yolo/train/images"
    
    print(f"📂 Scanning {train_dir}...")
    
    # We loop through all PNGs and decide based on filename
    for img_file in train_dir.glob("*.png"):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
            
        filename = img_file.name.lower()
        
        # LOGIC:
        # If it contains "bird", it's a bird.
        # If it contains "room", "wall", "floor", it's background.
        
        if "bird" in filename:
            images.append(img)
            labels.append("bird")
            # print(f"  + Added Bird: {filename}")
            
        elif any(x in filename for x in ["room", "wall", "floor", "empty"]):
            images.append(img)
            labels.append("background")
            # print(f"  - Added Background: {filename}")
    
    return images, labels

def main():
    # 1. Prepare Data
    print("🚀 Starting ResNet Training Pipeline...")
    images, labels = load_data()
    
    # Statistics
    n_bird = labels.count('bird')
    n_bg = labels.count('background')
    
    print(f"\n📊 Data Summary:")
    print(f"   - Total Images: {len(images)}")
    print(f"   - Birds (Positive): {n_bird}")
    print(f"   - Backgrounds (Negative): {n_bg}")
    
    if len(images) == 0:
        print("❌ Error: No images found. Check your 'data/artroom/bird/yolo/train/images' folder.")
        return

    # 2. Initialize Model
    detector = ResNetDetector()
    
    # 3. Train & Save
    detector.train_head(images, labels)
    
    # 4. Verification Test (Sanity Check)
    print("\n🔎 Running Sanity Check on Image 0...")
    lbl, conf, ms = detector.predict(images[0])
    print(f"   Result: {lbl} | Confidence: {conf:.2%} | Time: {ms:.2f}ms")
    print("\n✅ Training Complete. You can now use dev_03_test_resnet.ipynb")

if __name__ == "__main__":
    main()