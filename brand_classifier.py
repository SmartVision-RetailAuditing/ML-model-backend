from ultralytics import YOLO
from utils import urun_detay_bul

class BrandClassifier:
    def __init__(self, model_path, catalog_path, json_data):
        self.model = YOLO(model_path)
        self.katalog = json_data

# brand_classifier.py içinde bul ve değiştir:

    def classify(self, crop_rgb):
        # 'augment=True' kısmını sildik çünkü sınıflandırma modeli desteklemiyormuş.
        cls_res = self.model.predict(crop_rgb, verbose=False)[0] 
    
        idx = cls_res.probs.top1
        conf = cls_res.probs.top1conf.item()
        raw_name = cls_res.names[idx]
    
        from utils import urun_detay_bul
        marka, urun, conf = urun_detay_bul(raw_name, self.katalog)
        return marka, urun, conf