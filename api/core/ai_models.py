from ultralytics import YOLO
import easyocr
from api.utils.helpers import turkce_karakter_temizle, urun_detay_bul


class ShelfDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img, conf=0.45, iou=0.50):
        return self.model.predict(img, conf=conf, iou=iou, augment=True)[0]


class BrandClassifier:
    def __init__(self, model_path, katalog):
        self.model = YOLO(model_path)
        self.katalog = katalog

    def classify(self, crop_rgb):
        res = self.model.predict(crop_rgb, verbose=False, augment=True)[0]
        idx = res.probs.top1
        conf = res.probs.top1conf.item()
        raw_name = res.names[idx]

        marka, urun = urun_detay_bul(raw_name, self.katalog)
        return marka, urun, conf, raw_name


class TextValidator:
    def __init__(self, katalog):
        self.reader = easyocr.Reader(['tr', 'en'], gpu=False)
        self.markalar = set([k[1].get("brand").upper() for k in katalog.items() if k[1].get("brand")])

    def validate(self, crop_rgb):
        results = self.reader.readtext(crop_rgb, detail=1)
        for (bbox, text, prob) in results:
            txt = text.upper()
            if len(txt) < 3: continue

            for m in self.markalar:
                if m in txt and prob > 0.60:
                    return turkce_karakter_temizle(m.title()), 0.80
        return None, None