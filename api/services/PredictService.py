<<<<<<< HEAD
import os
import time
import cv2
import numpy as np
import json
import re
import random
import torch
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

print(f"🚀 Donanım Kontrolü: {'GPU (CUDA) Aktif!' if torch.cuda.is_available() else 'CPU Çalışıyor (GPU Bulunamadı)'}")
if torch.cuda.is_available():
    print(f"🔥 Ekran Kartı: {torch.cuda.get_device_name(0)}")

# --- .ENV'DEN GELEN DİNAMİK DEĞİŞKENLER ---
USE_AZURE = os.getenv("USE_AZURE", "False").lower() == "true"
GÜVEN_BARAJI = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.45"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.50"))
OCR_FALLBACK_THRESHOLD = float(os.getenv("OCR_FALLBACK_THRESHOLD", "0.15"))
EDGE_TOLERANCE = int(os.getenv("EDGE_TOLERANCE", "2"))

# Raf sınırlarını string olarak alıp integer (tam sayı) listesine çeviriyoruz
shelf_str = os.getenv("SHELF_THRESHOLDS", "400,800,1200,1600")
SHELF_THRESHOLDS = [int(x.strip()) for x in shelf_str.split(",")]

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

    # Yolo parametrelerini dinamik olarak yolluyoruz
    results = detector.detect(img, conf=YOLO_CONF, iou=YOLO_IOU)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if int(box.cls[0]) == 0: continue

        # Dinamik Kenar Toleransı
        if x1 <= EDGE_TOLERANCE or x2 >= (img_w - EDGE_TOLERANCE) or y1 <= EDGE_TOLERANCE or y2 >= (img_h - EDGE_TOLERANCE):
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        marka, urun, conf, raw_name = classifier.classify(crop_rgb)

        # Dinamik OCR Yedekleme (Fallback) Barajı
        if conf < OCR_FALLBACK_THRESHOLD or marka == "Bilinmiyor":
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

        # Dinamik Raf Yükseklik Barajları
        center_y = (y1 + y2) / 2
        if center_y < SHELF_THRESHOLDS[0]:
            raf_no = 1
        elif center_y < SHELF_THRESHOLDS[1]:
            raf_no = 2
        elif center_y < SHELF_THRESHOLDS[2]:
            raf_no = 3
        elif center_y < SHELF_THRESHOLDS[3]:
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
=======
import cv2
import json
import re
import torch
import pickle
import torch.nn.functional as F
from models.ai_models import ai_models
from utils.helpers import urun_detay_bul, turkce_karakter_temizle
from core import config


class PredictService:
    def __init__(
        self,
        catalog_path="product_catalog_sut.json",
        db_path="api/weights/dinov2_base_product_embeddings.pkl"
    ):
        # =========================
        # KATALOG
        # =========================
        with open(catalog_path, "r", encoding="utf-8") as f:
            self.katalog = json.load(f)

        self.katalog_markalar = set(
            k[1].get("brand", "").upper()
            for k in self.katalog.items()
        )

        # =========================
        # EMBEDDING DATABASE
        # =========================
        with open(db_path, "rb") as f:
            db_dict = pickle.load(f)

        self.db_names = list(db_dict.keys())
        self.db_vectors = torch.tensor(list(db_dict.values())).to(ai_models.device)

        # =========================
        # CLASS IDS
        # =========================
        self.names = ai_models.det_model.names
        self.tag_cls_id = 0

        for k, v in self.names.items():
            if any(word in v.lower() for word in ["price", "etiket", "tag"]):
                self.tag_cls_id = k

    # =========================================================
    # MAIN PIPELINE
    # =========================================================
    def process_image(self, img):
        img_h, img_w = img.shape[:2]

        products_list = []
        extracted_prices = []
        raw_products = []
        parsed_tags = []

        # =========================
        # YOLO DETECTION
        # =========================
        results = ai_models.det_model.predict(
            img,
            conf=config.YOLO_CONF,
            iou=config.YOLO_IOU,
            augment=False
        )[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])

            if cls_id == self.tag_cls_id:
                parsed_tags.append([x1, y1, x2, y2])
            else:
                raw_products.append({
                    "bbox": [x1, y1, x2, y2],
                    "box_obj": box
                })

        # =========================
        # OCR PRICE EXTRACTION
        # =========================
        for tx1, ty1, tx2, ty2 in parsed_tags:
            crop = img[ty1:ty2, tx1:tx2]
            if crop.size == 0:
                continue

            ocr_res = ai_models.ocr_reader.readtext(crop, detail=0)
            text = " ".join(ocr_res).upper().replace("O", "0")

            matches = re.findall(r"\d+[.,]\d+|\d+", text)

            tag_price = None
            for m in matches:
                try:
                    val = float(m.replace(",", "."))
                    if 0 < val < 5000:
                        tag_price = round(val, 2)
                        break
                except:
                    continue

            extracted_prices.append({
                "cx": (tx1 + tx2) / 2,
                "cy": (ty1 + ty2) / 2,
                "read_price": tag_price
            })

        # =========================
        # IMAGE EMBEDDING MATCH
        # =========================
        valid_products = []
        crop_tensors = []

        for prod in raw_products:
            x1, y1, x2, y2 = prod["bbox"]

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            tensor = ai_models.dino_preprocess(crop)
            crop_tensors.append(tensor)
            valid_products.append(prod)

        if crop_tensors:
            batch = torch.stack(crop_tensors)

            with torch.no_grad():
                feats = ai_models.dinov2_model(batch.to(ai_models.device))
                feats = F.normalize(feats, p=2, dim=1)

                sim = torch.mm(feats, self.db_vectors.T)
                top_scores, top_indices = torch.max(sim, dim=1)

            brand_price_map = {}
            temp_products = []

            for i, prod in enumerate(valid_products):
                x1, y1, x2, y2 = prod["bbox"]

                best_idx = top_indices[i].item()
                best_score = top_scores[i].item()

                raw_name = self.db_names[best_idx]
                details = urun_detay_bul(raw_name, self.katalog)
                marka = details.get("brand", raw_name) if details else "Bilinmiyor"

                # fallback OCR brand check
                if best_score < config.DINO_SIMILARITY_THRESHOLD:
                    marka = "Bilinmiyor"

                pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2

                # price matching
                ocr_price = None
                for tag in extracted_prices:
                    if tag["cy"] > pcy:
                        ocr_price = tag["read_price"]
                        break

                if ocr_price and marka != "Bilinmiyor":
                    brand_price_map[marka] = ocr_price

                temp_products.append({
                    "marka": marka,
                    "details": details,
                    "conf": best_score,
                    "bbox": [x1, y1, x2, y2],
                    "pcy": pcy,
                    "ocr_price": ocr_price
                })

            # =========================
            # FINAL DTO BUILD
            # =========================
            for tp in temp_products:
                marka = tp["marka"]
                details = tp["details"]

                final_price = tp["ocr_price"]

                if final_price is None and marka in brand_price_map:
                    final_price = brand_price_map[marka]

                if final_price is None:
                    try:
                        final_price = float(details.get("price"))
                    except:
                        final_price = None

                x1, y1, x2, y2 = tp["bbox"]

                products_list.append({
                    "product_code": details.get("original_code", "") if details else "",
                    "product_name": details.get("product_name", "Unknown"),
                    "brand_name": marka,
                    "volume": details.get("volume", "") if details else "",
                    "category": details.get("category", "") if details else "",
                    "price": final_price,
                    "confidence_score": round(float(tp["conf"]), 2),
                    "is_eye_level": 0.3 <= (tp["pcy"] / img_h) <= 0.6,
                    "shelf_position": self._get_shelf(tp["pcy"], img_h),
                    "bounding_box": {
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1
                    }
                })

        return {
            "products": products_list,
            "tags": extracted_prices
        }

    # =========================
    # HELPER
    # =========================
    def _get_shelf(self, y, img_h):
        ratio = y / img_h

        if ratio < 0.2:
            return 1
        elif ratio < 0.4:
            return 2
        elif ratio < 0.6:
            return 3
        elif ratio < 0.8:
            return 4
        return 5
>>>>>>> origin/development
