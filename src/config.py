# src/config.py
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
ARTROOM_DIR = DATA_DIR / "artroom"
BIRD_YOLO_DIR = ARTROOM_DIR / "bird" / "yolo"
BIRD_YAML = BIRD_YOLO_DIR / "bird_data.yaml"

# Model Paths
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATHS = {
    # 'yolo': MODEL_DIR / "yolov8n.pt",
    'yolo' : PROJECT_ROOT / "volov8n.pt", 
    'resnet': MODEL_DIR / "resnet18.pth",
    'resnet_head': MODEL_DIR / "resnet18_head.pkl",
    'mobilenet': MODEL_DIR / "mobilenet_v3.pth"
}

# Training Results
TRAINING_DIR = PROJECT_ROOT / "training"