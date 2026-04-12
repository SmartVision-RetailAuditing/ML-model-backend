import os
import time
import cv2
import numpy as np
import json
import re
import random
from dotenv import load_dotenv
from api.core.ai_models import ShelfDetector, BrandClassifier, TextValidator
from api.services.ImageService import upload_image_to_azure

load_dotenv()

# ==================================
# AYARLAR VE MODELLERİN YÜKLENMESİ
# ==================================
YOLO_DET_PATH = "models/best.pt"
YOLO_CLS_PATH = "models/best_large.pt"
CATALOG_PATH = "product_catalog_sut.json"
GÜVEN_BARAJI = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
USE_AZURE = os.getenv("USE_AZURE", "False").lower() == "true"
CIKTI_KLASORU = "ciktilar"

os.makedirs(CIKTI_KLASORU, exist_ok=True)

print("📂 Katalog yükleniyor...")
with open(CATALOG_PATH, 'r', encoding='utf-8') as f:
    katalog = json.load(f)

print("📦 Modeller belleğe alınıyor...")
detector = ShelfDetector(YOLO_DET_PATH)
classifier = BrandClassifier(YOLO_CLS_PATH, katalog)
validator = TextValidator(katalog)


def get_product_details(raw_name):
    clean_name = re.sub(r'(_v?\d+)$', '', raw_name, flags=re.IGNORECASE)

    def standartlastir(metin):
        if not metin: return ""
        metin = metin.replace('I', 'ı').replace('İ', 'i').lower()
        degisimler = {'ş': 's', 'ç': 'c', 'ğ': 'g', 'ü': 'u', 'ö': 'o', 'ı': 'i'}
        for tr, eng in degisimler.items():
            metin = metin.replace(tr, eng)
        return metin.strip()

    aranan_marka = standartlastir(clean_name)

    for key, details in katalog.items():
        katalogdaki_marka = standartlastir(details.get("brand", ""))
        katalogdaki_key = standartlastir(key)

        if katalogdaki_marka == aranan_marka or katalogdaki_key == aranan_marka:
            return details
    return None


def process_image(file) -> dict:
    t_basla = time.time()

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return {"azure_blob_url": None, "products": []}

    img_h, img_w = img.shape[:2]

    price_dict = {}
    detected_products = []

    results = detector.detect(img)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if int(box.cls[0]) == 0: continue

        tolerans = 2
        if x1 <= tolerans or x2 >= (img_w - tolerans) or y1 <= tolerans or y2 >= (img_h - tolerans):
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        marka, urun, conf, raw_name = classifier.classify(crop_rgb)

        if conf < 0.15 or marka == "Bilinmiyor":
            o_marka, o_conf = validator.validate(crop_rgb)
            if o_marka:
                marka = o_marka
                conf = o_conf
                raw_name = marka

        details = get_product_details(raw_name)

        original_code = details.get("original_code", raw_name) if details else raw_name
        product_name = details.get("product_name", urun) if details else urun
        volume = details.get("volume", "-") if details else "-"
        category = details.get("category", "BİLİNMİYOR") if details else "BİLİNMİYOR"

        if original_code not in price_dict:
            cat_price = details.get("price") if details else None
            if isinstance(cat_price, (int, float)):
                price_dict[original_code] = float(cat_price)
            else:
                price_dict[original_code] = random.randrange(3000, 10005, 5) / 100.0

        price_value = price_dict[original_code]

        center_y = (y1 + y2) / 2
        if center_y < 400:
            raf_no = 1
        elif center_y < 800:
            raf_no = 2
        elif center_y < 1200:
            raf_no = 3
        elif center_y < 1600:
            raf_no = 4
        else:
            raf_no = 5

        detected_products.append({
            "product_code": original_code,
            "product_name": product_name,
            "brand_name": marka,
            "volume": volume,
            "category": category,
            "price": price_value,
            "confidence_score": round(float(conf), 2),
            "is_eye_level": bool(raf_no in [3, 4, 5]),
            "shelf_position": raf_no,
            "bounding_box": {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1
            }
        })

        label = f"{marka} {price_value} TL"
        renk = (0, 255, 0) if conf >= GÜVEN_BARAJI else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), renk, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), renk, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    dosya_adi = f"analiz_{int(time.time())}.jpg"
    kayit_yolu = os.path.join(CIKTI_KLASORU, dosya_adi)
    cv2.imwrite(kayit_yolu, img)

    blob_url = upload_image_to_azure(kayit_yolu) if USE_AZURE else None

    t_bitis = time.time()
    print(f"🚀 Analiz Süresi: {t_bitis - t_basla:.3f} sn")

    return {
        "azure_blob_url": blob_url,
        "products": detected_products
    }