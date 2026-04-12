from ultralytics import YOLO
from utils import urun_detay_bul

class BrandClassifier:
    def __init__(self, model_path, catalog_path, json_data):
        self.model = YOLO(model_path)
        self.katalog = json_data

    def classify(self, crop_rgb):
        res = self.model.predict(crop_rgb, verbose=False, augment=True)[0]
        idx = res.probs.top1
        conf = res.probs.top1conf.item()
        raw_name = res.names[idx]
        
        marka, urun = urun_detay_bul(raw_name, self.katalog)
        return marka, urun, conf