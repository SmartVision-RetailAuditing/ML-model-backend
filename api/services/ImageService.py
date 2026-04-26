import cv2
import numpy as np
import uuid
import os
from azure.storage.blob import BlobServiceClient, ContentSettings
from core import config
from utils.helpers import turkce_karakter_temizle


class ImageService:
    @staticmethod
    def bytes_to_cv2(image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def draw_boxes_and_save(img, products, tags, output_path, baraj=config.DINO_SIMILARITY_THRESHOLD):
        # 1. Fiyat Etiketlerini Çiz (Mavi Kutular)
        for tag in tags:
            tx1, ty1, tx2, ty2 = tag["bbox"]
            val = tag.get("read_price")

            # Etiketin tam üzerine OCR ile okunan fiyatı yaz
            tag_text = f"{val:.2f} TL" if val is not None else "OKUNAMADI"
            cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (255, 120, 0), 2)
            cv2.putText(img, tag_text, (tx1, ty1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 120, 0), 1, cv2.LINE_AA)

        # 2. Ürünleri Çiz (Yeşil/Kırmızı Kutular ve Marka)
        for prod in products:
            bbox_info = prod.get("bounding_box")
            if not bbox_info: continue

            x1 = bbox_info.get("x", 0)
            y1 = bbox_info.get("y", 0)
            x2 = x1 + bbox_info.get("width", 0)
            y2 = y1 + bbox_info.get("height", 0)

            conf = prod.get("confidence_score", 0)
            marka = prod.get("brand_name", "Bilinmiyor")

            cizim_marka = turkce_karakter_temizle(marka)
            renk = (0, 255, 0) if conf >= baraj else (0, 0, 255)

            # --- GÜNCELLENEN KISIM: Fiyatı kaldırdık, sadece Marka ve Skor yazacak ---
            label = f"{cizim_marka} (%{int(conf * 100)})"

            # Kutuyu çiz
            cv2.rectangle(img, (x1, y1), (x2, y2), renk, 2)

            # Etiket arka planı ve Metni yaz
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), renk, -1)
            cv2.putText(img, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite(output_path, img)
        return output_path

    @staticmethod
    def upload_to_azure_bg(image_path: str, blob_name: str):
        if not config.USE_AZURE or not config.AZURE_BLOB_CONNECTION_STRING: return
        try:
            client = BlobServiceClient.from_connection_string(config.AZURE_BLOB_CONNECTION_STRING)
            container = client.get_container_client(config.AZURE_BLOB_CONTAINER_NAME)
            blob = container.get_blob_client(blob_name)
            with open(image_path, "rb") as data:
                blob.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type="image/jpeg"))
            print(f"☁️ Azure yüklemesi başarılı: {blob_name}")

            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"🧹 Lokal temizlik yapıldı: Kalabalık yapmaması için silindi.")
        except Exception as e:
            print(f"⚠️ Azure Hatası: {e}")