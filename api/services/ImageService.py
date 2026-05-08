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
        """
        Upload edilen image bytes -> OpenCV image
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def draw_boxes_and_save(img, products, tags, output_path, threshold=None):
        """
        Ürün kutularını çizer ve sonucu kaydeder
        """

        baraj = threshold or getattr(config, "DINO_SIMILARITY_THRESHOLD", 0.5)

        for prod in products:
            bbox = prod.get("bounding_box")
            if not bbox:
                continue

            x1 = bbox.get("x", 0)
            y1 = bbox.get("y", 0)
            x2 = x1 + bbox.get("width", 0)
            y2 = y1 + bbox.get("height", 0)

            conf = prod.get("confidence_score", 0)
            marka = prod.get("brand_name", "Bilinmiyor")

            marka_clean = turkce_karakter_temizle(marka)
            color = (0, 255, 0) if conf >= baraj else (0, 0, 255)

            label = f"{marka_clean} (%{int(conf * 100)})"

            # Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)

            # Label text
            cv2.putText(
                img,
                label,
                (x1, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        cv2.imwrite(output_path, img)
        return output_path

    @staticmethod
    def upload_to_azure_bg(image_path: str, blob_name: str):
        """
        Background task için Azure upload
        """

        if not getattr(config, "USE_AZURE", False):
            return None

        if not config.AZURE_BLOB_CONNECTION_STRING:
            return None

        try:
            blob_service = BlobServiceClient.from_connection_string(
                config.AZURE_BLOB_CONNECTION_STRING
            )

            container = blob_service.get_container_client(
                config.AZURE_BLOB_CONTAINER_NAME
            )

            blob_client = container.get_blob_client(blob_name)

            with open(image_path, "rb") as data:
                blob_client.upload_blob(
                    data,
                    overwrite=True,
                    content_settings=ContentSettings(content_type="image/jpeg")
                )

            print(f"☁️ Azure upload OK: {blob_name}")

            # local cleanup (opsiyonel)
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"🧹 Local file deleted: {image_path}")

        except Exception as e:
            print(f"❌ Azure upload error: {e}")

    @staticmethod
    def upload_to_azure_sync(image_path: str) -> str:
        """
        Sync upload (istersen direkt URL döndürmek için)
        """

        blob_name = f"{uuid.uuid4()}.jpg"

        blob_service = BlobServiceClient.from_connection_string(
            config.AZURE_BLOB_CONNECTION_STRING
        )

        container = blob_service.get_container_client(
            config.AZURE_BLOB_CONTAINER_NAME
        )

        blob_client = container.get_blob_client(blob_name)

        with open(image_path, "rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings=ContentSettings(content_type="image/jpeg")
            )

        account_name = config.AZURE_BLOB_CONNECTION_STRING.split("AccountName=")[1].split(";")[0]

        return f"https://{account_name}.blob.core.windows.net/{config.AZURE_BLOB_CONTAINER_NAME}/{blob_name}"