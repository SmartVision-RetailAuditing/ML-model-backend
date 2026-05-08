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