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
    def __init__(self, catalog_path="product_catalog_sut.json", db_path="weights/dinov2_large_product_embeddings.pkl"):
        with open(catalog_path, 'r', encoding='utf-8') as f:
            self.katalog = json.load(f)

        self.katalog_markalar = set([k[1].get("brand", "").upper() for k in self.katalog.items()])

        # --- DINOv2 Veritabanını GPU'ya Yükle ---
        with open(db_path, 'rb') as f:
            db_dict = pickle.load(f)

        self.db_names = list(db_dict.keys())
        # (N, 1024) boyutundaki dev matrisi tensöre çevirip ekran kartına alıyoruz
        self.db_vectors = torch.tensor(list(db_dict.values())).to(ai_models.device)
        # ------------------------------------------------------------

        # Dinamik Sınıf Tanıma: Modelden 'price_tag' sınıfının ID'sini bulur
        self.names = ai_models.det_model.names
        self.tag_cls_id = 0
        for k, v in self.names.items():
            if any(word in v.lower() for word in ['price', 'etiket', 'tag']):
                self.tag_cls_id = k

    def process_image(self, img):
        img_h, img_w = img.shape[:2]
        products_list = []

        # 1. YOLO TESPİTİ (Nesne Konumlarını Belirleme)
        results = ai_models.det_model.predict(
            img,
            conf=config.YOLO_CONF,
            iou=config.YOLO_IOU,
            augment=False
        )[0]

        raw_products = []
        parsed_tags = []

        # Kutuları Ayrıştır
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])

            if cls_id == self.tag_cls_id:
                parsed_tags.append([x1, y1, x2, y2])
            else:
                raw_products.append({"bbox": [x1, y1, x2, y2], "box_obj": box})

        # 2. ETİKETLERİ OKU (OCR - Fiyat Tespiti)
        extracted_prices = []
        for tx1, ty1, tx2, ty2 in parsed_tags:
            tag_crop = cv2.cvtColor(img[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2RGB)
            if tag_crop.size == 0: continue

            ocr_res = ai_models.ocr_reader.readtext(tag_crop, detail=0)
            text = " ".join(ocr_res).upper().replace('O', '0')

            matches = re.findall(r'\d+[.,]\d+|\d+', text)
            tag_price = None
            for match in matches:
                try:
                    val = float(match.replace(',', '.'))
                    if 0 < val < 2000:
                        tag_price = round(val, 2)
                        break
                except:
                    continue

            extracted_prices.append({
                "bbox": [tx1, ty1, tx2, ty2],
                "cx": (tx1 + tx2) / 2, "cy": (ty1 + ty2) / 2,
                "read_price": tag_price
            })

        # 3. ÜRÜNLERİ TOPLU İŞLE (BATCH INFERENCE DINOv2) - DARBOĞAZ ÇÖZÜMÜ EKLENDİ
        temp_products = []
        brand_price_map = {}

        valid_products = []
        crop_tensors = []

        # Önce tüm geçerli ürünleri kırpıp DINOv2 için tensöre çeviriyoruz
        for prod in raw_products:
            x1, y1, x2, y2 = prod["bbox"]
            if x1 <= config.EDGE_TOLERANCE or x2 >= (img_w - config.EDGE_TOLERANCE): continue

            crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            if crop.size == 0: continue

            # Resmi 518x518 yapıp tensöre çeviren işlem
            crop_tensor = ai_models.preprocess(crop)
            crop_tensors.append(crop_tensor)
            valid_products.append(prod)

        # Eğer sahnede ürün varsa hepsini parçalar (mini-batch) halinde karşılaştırıyoruz
        if crop_tensors:
            # Tensörleri önce RAM'de (CPU) istifle, GPU'yu hemen boğma
            batch_tensor = torch.stack(crop_tensors)
            all_features = []

            # VRAM kapasitene göre burayı ayarlayabilirsin.
            # RTX 3050 (4GB) için 2 veya 4 idealdir. Canavar sunucuda 16-32 yapabilirsin.
            chunk_size = 2

            with torch.no_grad():
                # Ürünleri chunk_size kadar gruplara bölerek GPU'ya al
                for i in range(0, len(batch_tensor), chunk_size):
                    mini_batch = batch_tensor[i:i + chunk_size].to(ai_models.device)

                    mini_features = ai_models.dinov2_model(mini_batch)
                    mini_features = F.normalize(mini_features, p=2, dim=1)

                    all_features.append(mini_features)

                # Çıkan tüm matematiksel sonuçları birleştir
                features = torch.cat(all_features, dim=0)

                # SİHİRLİ SATIR: Raftaki ürünlerle, veritabanındaki ürünleri matris çarpımı ile hızla karşılaştır
                similarity_scores = torch.mm(features, self.db_vectors.T)

                # Her ürün için en yüksek skoru ve bu skorun kime ait olduğunu (index) bul
                top_scores, top_indices = torch.max(similarity_scores, dim=1)

            # Sonuçları işle
            for i, prod in enumerate(valid_products):
                x1, y1, x2, y2 = prod["bbox"]

                # Eşleşen ürünün adını ve güven skorunu çek
                best_match_idx = top_indices[i].item()
                raw_name = self.db_names[best_match_idx]
                conf = top_scores[i].item()

                # İsimden _1, _2 gibi ekleri atıp katalogdan detayları getir
                details = urun_detay_bul(raw_name, self.katalog)
                marka = details.get("brand", raw_name) if details else "Bilinmiyor"

                # Düşük güvende OCR doğrulaması
                if conf < config.DINO_SIMILARITY_THRESHOLD or marka == "Bilinmiyor":
                    # Orijinal resmi tekrar alıp OCR'a gönder
                    crop_bgr = img[y1:y2, x1:x2]
                    ocr_val = ai_models.ocr_reader.readtext(crop_bgr, detail=1)
                    for (_, txt, prob) in ocr_val:
                        for m in self.katalog_markalar:
                            if m in txt.upper() and prob > 0.60:
                                marka, conf = turkce_karakter_temizle(m.title()), 0.80
                                break

                pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
                prod_h, prod_w = y2 - y1, x2 - x1

                # Fiyat eşleştirme
                ocr_price, min_dist = None, float('inf')
                for tag in extracted_prices:
                    if tag['cy'] > pcy and (tag['cy'] - pcy) < (prod_h * 2.5):
                        dist_x = abs(tag['cx'] - pcx)
                        if dist_x < min_dist and dist_x < (prod_w * 1.0):
                            min_dist, ocr_price = dist_x, tag['read_price']

                if ocr_price is not None and marka != "Bilinmiyor":
                    brand_price_map[marka] = ocr_price

                temp_products.append({
                    "marka": marka, "details": details, "conf": conf,
                    "ocr_price": ocr_price, "bbox": [x1, y1, x2, y2], "pcx": pcx, "pcy": pcy
                })

        # 4. Fiyat Paylaştır ve Final Listeyi Oluştur
        for tp in temp_products:
            marka, details = tp["marka"], tp["details"]
            final_price = tp["ocr_price"]

            if final_price is None and marka in brand_price_map:
                final_price = brand_price_map[marka]

            if final_price is None:
                try:
                    final_price = float(details.get("price"))
                except:
                    final_price = None

            # Raf İndeksi
            pcy = tp["pcy"]
            shelf_index = 1
            for threshold in config.SHELF_THRESHOLDS:
                if pcy > threshold:
                    shelf_index += 1
                else:
                    break

            # Bounding Box (X, Y, Width, Height)
            x1, y1, x2, y2 = tp["bbox"]
            bbox_dict = {
                "x": int(x1),
                "y": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1)
            }

            products_list.append({
                "product_code": details.get("original_code", "") if details else "",
                "product_name": details.get("product_name", "Bilinmeyen Ürün") if details else "Bilinmeyen Ürün",
                "brand_name": marka,
                "volume": details.get("volume", "") if details else "",
                "category": details.get("category", "") if details else "",
                "price": final_price,
                "confidence_score": round(float(tp["conf"]), 2),
                "is_eye_level": 0.3 <= (tp["pcy"] / img_h) <= 0.6,
                "shelf_position": shelf_index,
                "bounding_box": bbox_dict
            })

        return {"products": products_list, "tags": extracted_prices}