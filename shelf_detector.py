from ultralytics import YOLO

class ShelfDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        # TTA aktif: augment=True
        return self.model.predict(img, conf=0.45, iou=0.50, augment=True)[0]