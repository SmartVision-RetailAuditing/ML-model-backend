import cv2
import json
import re
from ultralytics import YOLO

# ==========================================
# AYARLAR
# ==========================================
YOLO_DET_PATH = "models/best.pt"
YOLO_CLS_PATH = "models/best_cls.pt"
CATALOG_PATH = "product_catalog_sut.json"        
TEST_IMAGE_PATH = "test_images/test_raf.jpg"
CIKTI_IMAGE_PATH = "sonuc_yolo_final_v3.jpg"      

# 189 sınıf için %40+ skor aslında çok güçlü bir tahmindir
GÜVEN_BARAJI = 0.25

def turkce_karakter_temizle(metin):
    if not metin: return metin
    degisim = {'ı':'i','İ':'I','I':'I','ğ':'g','Ğ':'G','ü':'u','Ü':'U','ş':'s','Ş':'S','ö':'o','Ö':'O','ç':'c','Ç':'C'}
    for tr, eng in degisim.items(): metin = metin.replace(tr, eng)
    return metin

def marka_bul(aranan_isim, katalog):
    # 1. Tam eşleşme denemesi
    if aranan_isim in katalog:
        return turkce_karakter_temizle(katalog[aranan_isim].get("brand", aranan_isim))
    
    # 2. Versiyon ve sayı temizleme (_v1, _1, _v2, _3 vb. hepsini atar)
    temiz_isim = re.sub(r'(_v?\d+)$', '', aranan_isim, flags=re.IGNORECASE)
    if temiz_isim in katalog:
        return turkce_karakter_temizle(katalog[temiz_isim].get("brand", temiz_isim))
    
    # 3. Parçalara ayırıp en uzun eşleşen markayı bulma
    # Örn: "153100000_ICIM_3" içindeki her parçayı JSON anahtarlarında ara
    for parca in aranan_isim.split("_"):
        if parca in katalog:
            return turkce_karakter_temizle(katalog[parca].get("brand", parca))
            
    # 4. Hiçbiri olmazsa en azından dosya adındaki marka kelimesini çek
    for p in aranan_isim.split("_"):
        if not p.isdigit() and len(p) > 2:
            return turkce_karakter_temizle(p)
            
    return "Bilinmiyor"

def main():
    print("🚀 Akıllı Eşleştirme Sistemi Başlatılıyor...")
    det_model = YOLO(YOLO_DET_PATH)
    cls_model = YOLO(YOLO_CLS_PATH)

    with open(CATALOG_PATH, 'r', encoding='utf-8') as f:
        katalog = json.load(f)

    img = cv2.imread(TEST_IMAGE_PATH)
    results = det_model.predict(img, conf=0.45, iou=0.50)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if int(box.cls[0]) == 0: continue # Etiketi atla

        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue
        
        # BGR -> RGB Çevrimi (Kritik!)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        cls_res = cls_model.predict(crop_rgb, verbose=False)[0]
        
        idx = cls_res.probs.top1
        conf = cls_res.probs.top1conf.item()
        raw_name = cls_res.names[idx]
        
        marka = marka_bul(raw_name, katalog)
        label = f"{marka} (%{int(conf*100)})"
        
        renk = (0, 255, 0) if conf >= GÜVEN_BARAJI else (0, 0, 255)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), renk, 2)
        cv2.putText(img, label, (x1, max(15, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
        cv2.putText(img, label, (x1, max(15, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, renk, 1)

    cv2.imwrite(CIKTI_IMAGE_PATH, img)
    print(f"✅ Analiz bitti: {CIKTI_IMAGE_PATH}")

if __name__ == "__main__":
    main()