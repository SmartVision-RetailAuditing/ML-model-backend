<<<<<<< HEAD
import easyocr
from utils import turkce_karakter_temizle

class TextValidator:
    def __init__(self, katalog):
        # Sadece ihtiyaç anında belleğe yüklenir
        self.reader = easyocr.Reader(['tr', 'en'], gpu=False)
        # Katalogdaki markaları bir listeye alalım
        self.markalar = set([k[1].get("brand").upper() for k in katalog.items()])

    def validate(self, crop_rgb):
        # detail=1 ile güven skoru da alırız
        results = self.reader.readtext(crop_rgb, detail=1)
        for (bbox, text, prob) in results:
            txt = text.upper()
            if len(txt) < 3: continue # "1L", "%3" gibi verileri ele
            
            for m in self.markalar:
                if m in txt and prob > 0.60: # Okuma güveni %60+ ise
                    return turkce_karakter_temizle(m.title()), 0.80 # OCR güvenini 0.80 say
        return None, None
=======
import cv2
import json
import torch
import pickle
import torch.nn.functional as F
import re
from rapidfuzz import fuzz, utils as fuzz_utils
from models.ai_models import ai_models
from utils.helpers import urun_detay_bul, turkce_karakter_temizle
from core import config


class PredictService:
    def __init__(self, catalog_path="product_catalog_sut.json", db_path="weights/dinov2_base_product_embeddings.pkl"):
        # Katalog Yükleme
        with open(catalog_path, 'r', encoding='utf-8') as f:
            self.katalog = json.load(f)

        self.katalog_markalar = set([k[1].get("brand", "").upper() for k in self.katalog.items()])

        # Vektör Veritabanı Yükleme (Eski, stabil DB'yi kullanıyoruz)
        with open(db_path, 'rb') as f:
            db_dict = pickle.load(f)

        self.db_names = list(db_dict.keys())
        self.db_vectors = torch.tensor(list(db_dict.values())).to(ai_models.device)

        self.names = ai_models.det_model.names
        self.tag_cls_id = 0
        for k, v in self.names.items():
            if any(word in v.lower() for word in ['price', 'etiket', 'tag']):
                self.tag_cls_id = k

    def create_clip_prompt(self, marka, details):
        kategori = details.get("category", "milk") if details else "milk"
        hacim = details.get("volume", "")
        return f"a supermarket shelf photo of {marka} brand {kategori} {hacim}, milk carton packaging"

    def process_image(self, img):
        img_h, img_w = img.shape[:2]
        products_list = []

        # 1. YOLO Nesne Tespiti
        results = ai_models.det_model.predict(img, conf=config.YOLO_CONF, iou=config.YOLO_IOU, augment=False)[0]

        raw_products, parsed_tags = [], []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if int(box.cls[0]) == self.tag_cls_id:
                parsed_tags.append([x1, y1, x2, y2])
            else:
                raw_products.append({"bbox": [x1, y1, x2, y2], "box_obj": box})

        # 2. Etiket Fiyatı Okuma
        extracted_prices = []
        for tx1, ty1, tx2, ty2 in parsed_tags:
            tag_crop = cv2.cvtColor(img[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2RGB)
            if tag_crop.size == 0: continue
            ocr_res = ai_models.ocr_reader.readtext(tag_crop, detail=0)
            text = " ".join(ocr_res).upper().replace('O', '0')
            matches = re.findall(r'\d+[.,]\d+|\d+', text)
            tag_price = next(
                (round(float(m.replace(',', '.')), 2) for m in matches if 0 < float(m.replace(',', '.')) < 2000), None)
            extracted_prices.append(
                {"bbox": [tx1, ty1, tx2, ty2], "cx": (tx1 + tx2) / 2, "cy": (ty1 + ty2) / 2, "read_price": tag_price})

        # 3. Kırpma (Eski, stabil, padding'siz yöntem)
        temp_products, brand_price_map = [], {}
        valid_products = [{"prod": p, "crop": cv2.cvtColor(img[p["bbox"][1]:p["bbox"][3], p["bbox"][0]:p["bbox"][2]],
                                                           cv2.COLOR_BGR2RGB)}
                          for p in raw_products if
                          p["bbox"][0] > config.EDGE_TOLERANCE and p["bbox"][2] < (img_w - config.EDGE_TOLERANCE)
                          and cv2.cvtColor(img[p["bbox"][1]:p["bbox"][3], p["bbox"][0]:p["bbox"][2]],
                                           cv2.COLOR_BGR2RGB).size > 0]

        if valid_products:
            all_features = []
            with torch.no_grad():
                for item in valid_products:
                    # TTA yok, düz çıkarım
                    t = ai_models.dino_preprocess(item["crop"]).unsqueeze(0).to(ai_models.device)
                    f = F.normalize(ai_models.dinov2_model(t), p=2, dim=1)
                    all_features.append(f)

                features = torch.cat(all_features, dim=0)
                # Basit ve stabil matris çarpımı (FAISS olmadan)
                similarity_scores = torch.mm(features, self.db_vectors.T)
                top_scores, top_indices = torch.topk(similarity_scores, min(3, similarity_scores.shape[1]), dim=1)

            # 4. Karar Aşaması
            for i, item in enumerate(valid_products):
                prod, crop_rgb = item["prod"], item["crop"]
                x1, y1, x2, y2 = prod["bbox"]

                candidates = []
                for j in range(top_indices.shape[1]):
                    idx, score = top_indices[i][j].item(), top_scores[i][j].item()
                    det = urun_detay_bul(self.db_names[idx], self.katalog)
                    candidates.append({"marka": det.get("brand", "Bilinmiyor"), "score": score, "details": det,
                                       "raw": self.db_names[idx]})

                best = candidates[0]
                f_name, f_marka, f_conf, f_details = best["raw"], best["marka"], best["score"], best["details"]

                # 🧠 KARARSIZLIK KONTROLÜ (Klasik Margin)
                margin = (candidates[0]["score"] - candidates[1]["score"]) / candidates[0]["score"] if len(
                    candidates) > 1 else 1.0

                # Eğer margin çok düşükse Hakemleri çağır
                if margin < 0.15 or best["score"] < 0.85:

                    # Basit OCR
                    pad = 5
                    px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
                    px2, py2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
                    ocr_res = ai_models.ocr_reader.readtext(img[py1:py2, px1:px2], detail=1)
                    ocr_text = " ".join([t for (_, t, p) in ocr_res if p > 0.4]).upper()

                    # Basit CLIP Prompt
                    prompts = [self.create_clip_prompt(c["marka"], c["details"]) for c in candidates]
                    inputs = ai_models.clip_processor(text=prompts, images=crop_rgb, return_tensors="pt",
                                                      padding=True).to(ai_models.device)
                    with torch.no_grad():
                        clip_probs = ai_models.clip_model(**inputs).logits_per_image.softmax(dim=1)[0]

                    best_f_score = -1.0
                    winner = candidates[0]

                    for idx, cand in enumerate(candidates):
                        ocr_s = fuzz.partial_ratio(cand["marka"].upper(), ocr_text,
                                                   processor=fuzz_utils.default_process) / 100.0
                        clip_s = clip_probs[idx].item()

                        # FÜZYON (Eski güvenilir formül)
                        cur_f = (0.5 * cand["score"]) + (0.2 * clip_s) + (0.3 * ocr_s)

                        # OCR %85 üstüyse direkt güven
                        if ocr_s > 0.85:
                            cur_f = (0.1 * cand["score"]) + (0.1 * clip_s) + (0.8 * ocr_s)

                        if cur_f > best_f_score:
                            best_f_score, winner = cur_f, cand

                    f_name, f_marka, f_conf, f_details = winner["raw"], winner["marka"], best_f_score, winner["details"]

                # Fiyat Eşleştirme
                pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
                prod_h, prod_w = y2 - y1, x2 - x1

                ocr_price, min_dist = None, float('inf')
                for tag in extracted_prices:
                    if tag['cy'] > pcy and (tag['cy'] - pcy) < (prod_h * 2.5):
                        dist_x = abs(tag['cx'] - pcx)
                        if dist_x < min_dist and dist_x < (prod_w * 1.0):
                            min_dist, ocr_price = dist_x, tag['read_price']

                if ocr_price is not None and f_marka != "Bilinmiyor":
                    brand_price_map[f_marka] = ocr_price

                temp_products.append({
                    "marka": f_marka, "details": f_details, "conf": f_conf,
                    "ocr_price": ocr_price, "bbox": [x1, y1, x2, y2], "pcx": pcx, "pcy": pcy
                })

        # 5. Sonuç Formatlama
        for tp in temp_products:
            marka, details, final_price, pcy = tp["marka"], tp["details"], tp["ocr_price"], tp["pcy"]
            if final_price is None and marka in brand_price_map: final_price = brand_price_map[marka]
            if final_price is None:
                try:
                    final_price = float(details.get("price"))
                except:
                    final_price = None

            shelf_index = 1
            for threshold in config.SHELF_THRESHOLDS:
                if pcy > threshold:
                    shelf_index += 1
                else:
                    break

            x1, y1, x2, y2 = tp["bbox"]
            products_list.append({
                "product_code": details.get("original_code", "") if details else "",
                "product_name": details.get("product_name", "Bilinmeyen Ürün") if details else "Bilinmeyen Ürün",
                "brand_name": marka, "volume": details.get("volume", "") if details else "",
                "category": details.get("category", "") if details else "", "price": final_price,
                "confidence_score": round(float(tp["conf"]), 2), "is_eye_level": 0.3 <= (pcy / img_h) <= 0.6,
                "shelf_position": shelf_index,
                "bounding_box": {"x": int(x1), "y": int(y1), "width": int(x2 - x1), "height": int(y2 - y1)}
            })

        return {"products": products_list, "tags": extracted_prices}
>>>>>>> origin/development
