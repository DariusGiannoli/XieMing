import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
import os
from src.config import MODEL_PATHS, TRAINING_DIR, BIRD_YAML

def run_fine_tuning(): 
    
    #Load model 
    model = YOLO('yolov8n.pt')
    # model = YOLO(MODEL_PATHS['yolo'])
    
    # Train the model
    results = model.train(
        data = BIRD_YAML,
        epochs = 50, 
        imgsz = 640, 
        batch = 4, 
        name = 'bird_artroom_finetune', 
        project = str(TRAINING_DIR / "runs"),
        exist_ok = False
    )
    
    print("Training completed successfully!")
    
if __name__ == "__main__": 
    run_fine_tuning()