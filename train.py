from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Train model
model.train(
    data="dataset/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    name="ppe_model"
)