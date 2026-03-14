import cv2
import pickle
import torch
import numpy as np
import json
import re                                                
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

# ==========================================
# TÜRKÇE KARAKTER TEMİZLEYİCİ FONKSİYON
# ==========================================
def turkce_karakter_temizle(metin):
    if not metin: return metin
    degisim = {
        'ı': 'i', 'İ': 'I', 'I': 'I',
        'ğ': 'g', 'Ğ': 'G',
        'ü': 'u', 'Ü': 'U',
        'ş': 's', 'Ş': 'S',
        'ö': 'o', 'Ö': 'O',
        'ç': 'c', 'Ç': 'C'
    }
    for tr, eng in degisim.items():
        metin = metin.replace(tr, eng)
    return metin

# ==========================================
# MARKA BULUCU FONKSİYON (Senin kodundan paketlendi)
# ==========================================
def marka_bul(aranan_isim, katalog):
    sade_marka = None
    if aranan_isim in katalog:
        sade_marka = katalog[aranan_isim].get("brand")
    if not sade_marka:
        deneme_isim = re.sub(r'_v\d+$', '', aranan_isim, flags=re.IGNORECASE)
        if deneme_isim in katalog:
            sade_marka = katalog[deneme_isim].get("brand")
    if not sade_marka:
        for p in aranan_isim.split("_"):
            if not p.isdigit() and not re.match(r'^v\d+$', p, re.IGNORECASE):
                sade_marka = p
                break
    if not sade_marka:
        sade_marka = "Bilinmeyen_Marka"
    
    return turkce_karakter_temizle(sade_marka)

# ==========================================
# AYARLAR
# ==========================================
YOLO_MODEL_PATH = "models/best.pt"
DATABASE_PATH = "product_embeddings.pkl"
CATALOG_PATH = "product_catalog_sut.json"        
TEST_IMAGE_PATH = "test_images/test_raf.jpg"
CIKTI_IMAGE_PATH = "sonuc_resnet_temiz.jpg"      

BENZERLIK_BARAJI = 0.70 # %70 ve üzeri benzerlikleri kabul et

def main():
    print("🚀 ResNet50 + JSON Raf Analiz Sistemi Başlatılıyor...\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Modelleri Yükle
    print("1/4: YOLO Yükleniyor...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    print("2/4: ResNet50 Özellik Çıkarıcı Yükleniyor...")
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    resnet.to(device)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Hafızayı (Veritabanını) Yükle
    print("3/4: Ürün Parmak İzi Hafızası Yüklendi.")
    try:
        with open(DATABASE_PATH, 'rb') as f:
            veritabani = pickle.load(f)
    except FileNotFoundError:
        print(f"HATA: '{DATABASE_PATH}' bulunamadı. Önce database_builder.py'yi çalıştırın!")
        return
        
    # 3. JSON Kataloğunu Yükle
    print("4/4: Ürün Kataloğu (JSON) Yükleniyor...")
    try:
        with open(CATALOG_PATH, 'r', encoding='utf-8') as f:
            katalog = json.load(f)
    except Exception as e:
        print(f"HATA: '{CATALOG_PATH}' okunamadı: {e}")
        return

    # 4. Görüntüyü Oku ve Analiz Et
    print("\nRaf analiz ediliyor, lütfen bekleyin...")
    orijinal_resim = cv2.imread(TEST_IMAGE_PATH)
    if orijinal_resim is None:
        print("HATA: Test resmi bulunamadı!")
        return

    sonuclar = yolo_model.predict(orijinal_resim, conf=0.45, iou=0.50)[0]
    
    toplam_kutu = len(sonuclar.boxes)
    print(f"Rafta toplam {toplam_kutu} nesne bulundu. Tanımlama başlıyor...\n")

    kutu_sayaci = 0

    for box_data in sonuclar.boxes:
        x1, y1, x2, y2 = map(int, box_data.xyxy[0])
        sinif_id = int(box_data.cls[0]) 

        # Fiyat etiketini atla
        if sinif_id == 0: 
            cv2.rectangle(orijinal_resim, (x1, y1), (x2, y2), (0, 255, 255), 2)
            continue 

        crop_img = orijinal_resim[y1:y2, x1:x2]
        if crop_img.shape[0] < 30 or crop_img.shape[1] < 30: continue
        
        kutu_sayaci += 1
        
        # Kesilen raf görüntüsünü ResNet50 için hazırla
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        # Raftaki kutunun parmak izini çıkar
        with torch.no_grad():
            crop_features = resnet(img_tensor).flatten()
            crop_features = crop_features / crop_features.norm(p=2)
            crop_features = crop_features.cpu().numpy()

        # --- BÜTÜN İHTİMALLERİ HESAPLA VE LİSTELE ---
        skorlar = []
        for urun_adi, db_features in veritabani.items():
            skor = np.dot(crop_features, db_features)
            skorlar.append((skor, urun_adi))
            
        # Skorları büyükten küçüğe sırala
        skorlar.sort(key=lambda x: x[0], reverse=True)

        # 1. KONSOLA YAZDIRMA (En iyi 3 İhtimal)
        print(f"--- Kutu {kutu_sayaci}/{toplam_kutu} Analizi ---")
        for i in range(min(3, len(skorlar))):
            s_skor, s_urun = skorlar[i]
            s_marka = marka_bul(s_urun, katalog)
            print(f"  {i+1}. İhtimal: {s_marka} (%{int(s_skor*100)})")
        print("-" * 30)

        # 2. GÖRÜNTÜYE YAZDIRMA (Sadece 1. İhtimal)
        en_iyi_skor, en_iyi_eslesme_adi = skorlar[0]

        if en_iyi_skor >= BENZERLIK_BARAJI:
            renk = (0, 255, 0) # YEŞİL
            sade_marka = marka_bul(en_iyi_eslesme_adi, katalog)
            yazi = f"{sade_marka} (%{int(en_iyi_skor*100)})"
        else:
            renk = (0, 0, 255) # KIRMIZI
            yazi = f"Bilinmiyor (%{int(en_iyi_skor*100)})"

        cv2.rectangle(orijinal_resim, (x1, y1), (x2, y2), renk, 2)
        
        cv2.putText(orijinal_resim, yazi, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3) 
        cv2.putText(orijinal_resim, yazi, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, renk, 1)    

    cv2.imwrite(CIKTI_IMAGE_PATH, orijinal_resim)
    print(f"\n✅ İşlem Tamamlandı! Lütfen '{CIKTI_IMAGE_PATH}' dosyasına bakarak sonucu incele.")

if __name__ == "__main__":
    main()