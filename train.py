import kagglehub
from ultralytics import YOLO
import os

# Download the dataset using kagglehub (exact method as provided)
path = kagglehub.dataset_download("vinayakshanawad/weedcrop-image-dataset")

# Define dataset.yaml file path
import yaml
from pathlib import Path

# Update paths in data.yaml dynamically
original_yaml = Path(path) / "WeedCrop.v1i.yolov5pytorch" / "data.yaml"

with open(original_yaml, 'r') as f:
    data = yaml.safe_load(f)

# Adjust paths to absolute paths
data_yaml = {
    'train': str(Path(path) / "WeedCrop.v1i.yolov5pytorch" / "train" / "images"),
    'val': str(Path(path) / "WeedCrop.v1i.yolov5pytorch" / "valid" / "images"),
    'test': str(Path(path) / "WeedCrop.v1i.yolov5pytorch" / "test" / "images"),
    'nc': data['nc'],
    'names': data['names']
}

# Save updated data.yaml locally
updated_yaml_path = Path("updated_data.yaml")
with open(updated_yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

# Load YOLOv8 model (choose your version, e.g., yolov8n.pt for nano, yolov8s.pt for small)
model = YOLO('yolov8n.pt')

# Start training
model.train(
    data=str(updated_yaml_path),
    epochs=50,
    batch=16,
    imgsz=640,
    device='0', # GPU
)

# Validate the trained model
metrics = model.val()

# Save the trained model
model.export(format='pt', name='weed_detection_model.pt')

# Print metrics
print(metrics)
