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


# ================== YARDIMCI FONKSİYON ==================
def turkce_karakter_temizle(metin):
    if not metin: return metin
    degisim = {'ı': 'i', 'İ': 'I', 'I': 'I', 'ğ': 'g', 'Ğ': 'G', 'ü': 'u', 'Ü': 'U',
               'ş': 's', 'Ş': 'S', 'ö': 'o', 'Ö': 'O', 'ç': 'c', 'Ç': 'C'}
    for tr, eng in degisim.items():
        metin = metin.replace(tr, eng)
    return metin


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
    price_dict = {}

    category_counts = {}
    product_counts = {}
    detected_items = []

    # ------------------ 1. AŞAMA: TESPİT VE SAYIM ------------------
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

        # ========== AKILLI KATALOG ARAMA EKLENDİ ==========
        if best_score >= SIMILARITY_THRESHOLD and best_match:
            tam_kod = best_match
            info = None

            # 1. Birebir eşleşme var mı?
            if tam_kod in katalog:
                info = katalog[tam_kod].copy()
            else:
                # 2. Sonundaki _v2, _v3, _1 gibi versiyon eklerini silip dene
                temiz_kod = re.sub(r'(_v?\d+)$', '', tam_kod, flags=re.IGNORECASE)
                if temiz_kod in katalog:
                    info = katalog[temiz_kod].copy()
                else:
                    # 3. Hala bulamadıysa alt çizgiye göre bölüp sadece ilk kısmı (ör: 153106322) ara
                    barkod_kismi = tam_kod.split('_')[0]
                    if barkod_kismi in katalog:
                        info = katalog[barkod_kismi].copy()

            # Hiçbir formatta katalogda bulamadıysa varsayılan atama
            if info is None:
                info = {
                    "original_code": tam_kod,
                    "brand": "Bilinmiyor",
                    "product_name": "Katalogda Bulunamadı",
                    "volume": "-",
                    "category": "SÜT"
                }

            info["similarity"] = round(best_score, 2)
            urun_kodu = best_match
            kutu_renk = (0, 255, 0)
        else:
            info = {
                "original_code": "UNKNOWN",
                "brand": "Bilinmiyor",
                "product_name": "Tanımsız Ürün",
                "volume": "-",
                "category": "BİLİNMİYOR",
                "similarity": round(best_score, 2)
            }
            urun_kodu = "Bilinmeyen_Urun"
            kutu_renk = (0, 0, 255)

        # Fiyat Hesaplama
        if urun_kodu not in price_dict:
            price_dict[urun_kodu] = random.randrange(3000, 10005, 5) / 100.0
        info["price"] = price_dict[urun_kodu]

        # Raf ve Göz Hizası
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

        info["shelf_number"] = raf_no
        info["eye_level"] = True if raf_no in [3, 4] else False

        # Sayaçları artır
        kategori = info.get("category", "BİLİNMİYOR")
        category_counts[kategori] = category_counts.get(kategori, 0) + 1
        product_counts[urun_kodu] = product_counts.get(urun_kodu, 0) + 1

        detected_items.append({
            "urun_kodu": urun_kodu,
            "info": info,
            "kutu_renk": kutu_renk,
            "box_coords": (x1, y1, x2, y2),
            "best_score": best_score,
            "raf_no": raf_no
        })

    # ------------------ 2. AŞAMA: ORAN HESAPLAMA VE ÇIKTI ------------------
    for item in detected_items:
        urun_kodu = item["urun_kodu"]
        info = item["info"]
        kategori = info.get("category", "BİLİNMİYOR")

        toplam_kategori_urunu = category_counts[kategori]
        urun_adeti = product_counts[urun_kodu]

        yuzdelik_oran = (urun_adeti / toplam_kategori_urunu) * 100 if toplam_kategori_urunu > 0 else 0
        info["shelf_share_percentage"] = round(yuzdelik_oran, 2)

        x1, y1, x2, y2 = item["box_coords"]
        info["coordinates"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        # Görsel üzerine çizim
        kutu_renk = item["kutu_renk"]
        best_score = item["best_score"]
        raf_no = item["raf_no"]

        cv2.rectangle(output_img, (x1, y1), (x2, y2), kutu_renk, 2)

        # Sadece bilinen markaları ekrana yazdırır
        gorunen_marka = info.get("brand", "Bilinmiyor")
        cv2.putText(output_img, f"{gorunen_marka} ({int(best_score * 100)}%) - Raf:{raf_no} - %{int(yuzdelik_oran)}",
                    (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, kutu_renk, 1)

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