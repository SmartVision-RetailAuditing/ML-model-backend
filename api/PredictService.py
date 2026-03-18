import os
import time
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import pickle, json
from api.ImageService import upload_image_to_azure
from dotenv import load_dotenv
import re

load_dotenv()

# ==================================
# AYARLAR
# ==================================
YOLO_MODEL_PATH = "api/models/best.pt"
YOLO_CLS_PATH = "api/models/best_cls.pt"  # Opsiyonel: marka sınıflandırıcı
DATABASE_PATH = "api/product_embeddings.pkl"
CATALOG_PATH = "api/product_catalog_sut.json"

# DINOv2 için benzerlik eşiği yükseltildi.
# Testlerinize göre 0.60 ile 0.80 arasında bir değere ince ayar yapabilirsiniz.
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

# 1. YOLO Modelleri
print("📦 YOLO Modelleri yükleniyor...")
yolo_model = YOLO(YOLO_MODEL_PATH)
cls_model = YOLO(YOLO_CLS_PATH)

# 2. DINOv2 Modeli (ResNet yerine)
print("🧠 DINOv2 Modeli yükleniyor...")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.eval()
dinov2.to(device)

# DINOv2 için Görüntü Ön İşleme Pipeline'ı
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), # DINOv2'nin patch boyutu olan 14'e tam bölünür
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Veritabanı ve Katalog Yükleme
print("📂 Veritabanı ve Katalog yükleniyor...")
with open(DATABASE_PATH, 'rb') as f:
    veritabani = pickle.load(f)
with open(CATALOG_PATH, 'r', encoding='utf-8') as f:
    katalog = json.load(f)

# ================== YARDIMCI FONKSİYON ==================
def turkce_karakter_temizle(metin):
    if not metin: return metin
    degisim = {'ı':'i','İ':'I','I':'I','ğ':'g','Ğ':'G','ü':'u','Ü':'U',
               'ş':'s','Ş':'S','ö':'o','Ö':'O','ç':'c','Ç':'C'}
    for tr, eng in degisim.items():
        metin = metin.replace(tr, eng)
    return metin

def marka_bul(aranan_isim, katalog):
    if aranan_isim in katalog:
        return turkce_karakter_temizle(katalog[aranan_isim].get("brand", aranan_isim))
    temiz_isim = re.sub(r'(_v?\d+)$', '', aranan_isim, flags=re.IGNORECASE)
    if temiz_isim in katalog:
        return turkce_karakter_temizle(katalog[temiz_isim].get("brand", temiz_isim))
    for parca in aranan_isim.split("_"):
        if parca in katalog:
            return turkce_karakter_temizle(katalog[parca].get("brand", parca))
    for p in aranan_isim.split("_"):
        if not p.isdigit() and len(p) > 2:
            return turkce_karakter_temizle(p)
    return "Bilinmiyor"

# ================== ANA FONKSİYON ==================
def process_image(file) -> dict:
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    output_img = img_cv.copy()

    results = yolo_model.predict(img_cv, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ)[0]
    if results.boxes is None:
        return {"total_count": 0, "azure_blob_url": None, "products": {}}

    sonuclar_dict = {}
    for box in results.boxes:
        sinif_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Fiyat kutularını atla
        if sinif_id == PRICE_CLASS_ID:
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            continue

        crop = img_cv[y1:y2, x1:x2]
        CROP_DIR = "crops"
        os.makedirs(CROP_DIR, exist_ok=True)

        crop_name = f"crop_{x1}_{y1}.jpg"
        cv2.imwrite(os.path.join(CROP_DIR, crop_name), crop)

        # ========== DINOv2 EMBEDDING ==========
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = preprocess(pil_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            # ResNet yerine DINOv2 kullanılarak parmak izi (384 boyutlu vektör) çıkarılır
            features = dinov2(tensor).flatten()
            features = features / features.norm(p=2)
            features = features.cpu().numpy()

        # ========== VERİTABANI BENZERLİĞİ ==========
        best_score = 0
        best_match = None
        for product_code, db_feat in veritabani.items():
            score = np.dot(features, db_feat)
            if score > best_score:
                best_score = float(score)
                best_match = product_code

        # ========== BENZERLİK YETERLİYSE ==========
        if best_score >= SIMILARITY_THRESHOLD and best_match:
            # BURAYI DEĞİŞTİRİYORUZ: Artık kesme işlemi yapmıyoruz, tam adı alıyoruz.
            tam_kod = best_match

            info = katalog.get(tam_kod, {
                "original_code": tam_kod,
                "brand": "Bilinmiyor",
                "product_name": "Katalogda Bulunamadı",
                "volume": "-",
                "category": "SÜT"
            }).copy()

            info["similarity"] = round(best_score, 2)
            info["source"] = "embedding"
            urun_kodu = best_match
            kutu_renk = (0, 255, 0)
        else:
            info = {
                "original_code": "UNKNOWN",
                "brand": "Bilinmiyor",
                "product_name": "Tanımsız Ürün",
                "volume": "-",
                "category": "BİLİNMİYOR",
                "similarity": round(best_score, 2),
                "source": "embedding"
            }
            urun_kodu = "Bilinmeyen_Urun"
            kutu_renk = (0, 0, 255)

        info["price"] = 23.5
        info["coordinates"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        cv2.rectangle(output_img, (x1, y1), (x2, y2), kutu_renk, 2)
        cv2.putText(output_img, f"{info['brand']} ({int(best_score*100)}%)",
                    (x1, max(15, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, kutu_renk, 1)

        key = urun_kodu
        sayac = 1
        while key in sonuclar_dict:
            key = f"{urun_kodu}_{sayac}"
            sayac += 1
        sonuclar_dict[key] = info

    # ================== ÇIKTI ==================
    dosya_adi = f"analiz_{int(time.time())}.jpg"
    kayit_yolu = os.path.join(CIKTI_KLASORU, dosya_adi)
    cv2.imwrite(kayit_yolu, output_img)

    blob_url = None
    if USE_AZURE:
        blob_url = upload_image_to_azure(kayit_yolu)

    return {"total_count": len(sonuclar_dict), "azure_blob_url": blob_url, "products": sonuclar_dict}