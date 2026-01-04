from ultralytics import YOLO

# load a pretrained model (recommended for training)
model = YOLO("best.pt")

# Train the model
results = model.train(data="data.yaml", epochs=10, imgsz=640)
