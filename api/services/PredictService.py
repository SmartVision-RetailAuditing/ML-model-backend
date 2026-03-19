import os
import time
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import pickle, json
from api.services.ImageService import upload_image_to_azure
from dotenv import load_dotenv
import re
import random

load_dotenv()

# ==================================
# AYARLAR
# ==================================
YOLO_MODEL_PATH = "api/models/best.pt"
YOLO_CLS_PATH = "api/models/best_cls.pt"
DATABASE_PATH = "api/product_embeddings.pkl"
CATALOG_PATH = "api/product_catalog_sut.json"

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.40"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.50"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.50"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))

CIKTI_KLASORU = "ciktilar"
USE_AZURE = os.getenv("USE_AZURE", "False").lower() == "true"
PRODUCT_CLASS_ID = 1
PRICE_CLASS_ID = 0

if not os.path.exists(CIKTI_KLASORU):
    os.makedirs(CIKTI_KLASORU)

# ================== MODELLER ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Kullanılan İşlemci: {device}")

print("📦 YOLO Modelleri yükleniyor...")
yolo_model = YOLO(YOLO_MODEL_PATH)
cls_model = YOLO(YOLO_CLS_PATH)

print("🧠 DINOv2 Modeli yükleniyor...")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.eval()
dinov2.to(device)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("📂 Veritabanı ve Katalog yükleniyor...")
with open(DATABASE_PATH, 'rb') as f:
    veritabani = pickle.load(f)
with open(CATALOG_PATH, 'r', encoding='utf-8') as f:
    katalog = json.load(f)


# ================== ANA FONKSİYON ==================
def process_image(file) -> dict:
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    output_img = img_cv.copy()

    results = yolo_model.predict(img_cv, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ)[0]

    # Hiçbir şey bulunamazsa boş liste dön
    if results.boxes is None:
        return {"products": []}

    price_dict = {}
    detected_items = []
    final_products_list = []

    # ------------------ 1. AŞAMA: TESPİT VE BİLGİ TOPLAMA ------------------
    for box in results.boxes:
        sinif_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if sinif_id == PRICE_CLASS_ID:
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            continue

        crop = img_cv[y1:y2, x1:x2]
        CROP_DIR = "crops"
        os.makedirs(CROP_DIR, exist_ok=True)

        crop_name = f"crop_{x1}_{y1}.jpg"
        cv2.imwrite(os.path.join(CROP_DIR, crop_name), crop)

        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = preprocess(pil_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            features = dinov2(tensor).flatten()
            features = features / features.norm(p=2)
            features = features.cpu().numpy()

        best_score = 0
        best_match = None
        for product_code, db_feat in veritabani.items():
            score = np.dot(features, db_feat)
            if score > best_score:
                best_score = float(score)
                best_match = product_code

        # Akıllı Katalog Arama
        if best_score >= SIMILARITY_THRESHOLD and best_match:
            tam_kod = best_match
            katalog_info = None

            # 1. Birebir eşleşme
            if tam_kod in katalog:
                katalog_info = katalog[tam_kod].copy()
            else:
                # 2. Versiyon eklerini silip arama
                temiz_kod = re.sub(r'(_v?\d+)$', '', tam_kod, flags=re.IGNORECASE)
                if temiz_kod in katalog:
                    katalog_info = katalog[temiz_kod].copy()
                else:
                    # 3. Sadece barkod kısmıyla arama
                    barkod_kismi = tam_kod.split('_')[0]
                    if barkod_kismi in katalog:
                        katalog_info = katalog[barkod_kismi].copy()

            if katalog_info is None:
                katalog_info = {
                    "brand": "Bilinmiyor",
                    "product_name": "Katalogda Bulunamadı",
                    "volume": "-",
                    "category": "SÜT"
                }

            urun_kodu = best_match
            kutu_renk = (0, 255, 0)
        else:
            katalog_info = {
                "brand": "Bilinmiyor",
                "product_name": "Tanımsız Ürün",
                "volume": "-",
                "category": "BİLİNMİYOR"
            }
            urun_kodu = "Bilinmeyen_Urun"
            kutu_renk = (0, 0, 255)

        # Dinamik ve Tutarlı Fiyat Hesaplama
        if urun_kodu not in price_dict:
            price_dict[urun_kodu] = random.randrange(3000, 10005, 5) / 100.0

        # Raf Tahmini
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

        detected_items.append({
            "urun_kodu": urun_kodu,
            "katalog_info": katalog_info,
            "kutu_renk": kutu_renk,
            "box_coords": (x1, y1, x2, y2),
            "best_score": best_score,
            "raf_no": raf_no
        })

    # ------------------ 2. AŞAMA: JSON OBJESİNİ (DTO) OLUŞTURMA ------------------
    for item in detected_items:
        urun_kodu = item["urun_kodu"]
        katalog_info = item["katalog_info"]
        x1, y1, x2, y2 = item["box_coords"]
        raf_no = item["raf_no"]
        best_score = item["best_score"]

        # C# tarafındaki DTO'nun [JsonPropertyName] etiketlerine tam uyumlu yapı
        product_dto = {
            "product_code": urun_kodu,
            "product_name": katalog_info.get("product_name", "Tanımsız Ürün"),
            "brand_name": katalog_info.get("brand", "Bilinmiyor"),
            "volume": katalog_info.get("volume", "-"),
            "category": katalog_info.get("category", "BİLİNMİYOR"),
            "price": price_dict[urun_kodu],
            "confidence_score": round(best_score, 2),
            "is_eye_level": bool(raf_no in [3, 4, 5]),
            "shelf_position": raf_no,
            "bounding_box": {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1
            }
        }

        final_products_list.append(product_dto)

        # Görsel üzerine çizim (opsiyonel hata ayıklama / görüntüleme için)
        kutu_renk = item["kutu_renk"]
        gorunen_marka = katalog_info.get("brand", "Bilinmiyor")
        cv2.rectangle(output_img, (x1, y1), (x2, y2), kutu_renk, 2)
        cv2.putText(output_img, f"{gorunen_marka} ({int(best_score * 100)}%) - Raf:{raf_no}",
                    (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, kutu_renk, 1)

    # ================== ÇIKTI KAYDEDİLİYOR ==================
    dosya_adi = f"analiz_{int(time.time())}.jpg"
    kayit_yolu = os.path.join(CIKTI_KLASORU, dosya_adi)
    cv2.imwrite(kayit_yolu, output_img)

    blob_url = None
    if USE_AZURE:
        blob_url = upload_image_to_azure(kayit_yolu)

    # C#'ın AiVisionResultDto nesnesinin tam beklediği format
    return {
        "azure_blob_url": blob_url,  # C# tarafında DTO'ya eklersen okur, eklemezsen ignore eder.
        "products": final_products_list
    }