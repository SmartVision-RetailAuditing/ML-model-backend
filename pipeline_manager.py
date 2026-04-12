import cv2
import os
import glob
import logging
import json
import re

# Konsoldaki gereksiz kütüphane loglarını susturur
logging.getLogger('ultralytics').setLevel(logging.ERROR)

class PipelineManager:
    def __init__(self, det_p, cls_p, cat_p, test_d, baraj, output_d="output_images"):
        # Kataloğu yükle
        with open(cat_p, 'r', encoding='utf-8') as f:
            self.katalog = json.load(f)
            
        self.test_dir = test_d
        self.output_dir = output_d
        self.baraj = baraj
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        from shelf_detector import ShelfDetector
        from brand_classifier import BrandClassifier
        from text_validator import TextValidator
        
        self.detector = ShelfDetector(det_p)
        self.classifier = BrandClassifier(cls_p, cat_p, self.katalog)
        self.validator = TextValidator(self.katalog)

    def get_product_details(self, raw_name):
        """Katalogdan ürün detaylarını çeker (Türkçe Karakter ve Büyük/Küçük Harf Duyarsız)"""
        import re
        
        # 1. AI'dan gelen raw_name'i temizle (Örn: Sutas_v1 -> Sutas)
        clean_name = re.sub(r'(_v?\d+)$', '', raw_name, flags=re.IGNORECASE)
        
        # 2. Karakterleri eşitlemek için bir iç fonksiyon
        def standartlastir(metin):
            if not metin: return ""
            metin = metin.replace('I','ı').replace('İ','i').lower()
            degisimler = {'ş':'s', 'ç':'c', 'ğ':'g', 'ü':'u', 'ö':'o', 'ı':'i'}
            for tr, eng in degisimler.items():
                metin = metin.replace(tr, eng)
            return metin.strip()

        aranan_marka = standartlastir(clean_name)

        # 3. Katalogda Ara
        for key, details in self.katalog.items():
            katalogdaki_marka = standartlastir(details.get("brand", ""))
            katalogdaki_key = standartlastir(key)
            
            # Eğer AI'ın bulduğu "sutas", katalogdaki "sutas" ile eşleşirse o detayı çek
            if katalogdaki_marka == aranan_marka or katalogdaki_key == aranan_marka:
                return details
        
        # Katalogda gerçekten hiçbir şekilde yoksa None dön
        return None

    def process_image(self, resim_yolu):
        dosya_adi = os.path.basename(resim_yolu)
        print(f"📸 İşleniyor: {dosya_adi}")
        
        img = cv2.imread(resim_yolu)
        if img is None: return

        img_h, img_w = img.shape[:2]
        
        # Panel için JSON listesi başlat
        image_results = {
            "fileName": dosya_adi,
            "detectedProducts": []
        }

        results = self.detector.detect(img)

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if int(box.cls[0]) == 0: continue 

            tolerans = 2
            if x1 <= tolerans or x2 >= (img_w - tolerans) or y1 <= tolerans or y2 >= (img_h - tolerans):
                continue 

            crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            marka, urun, conf = self.classifier.classify(crop)
            
            if conf < 0.15 or marka == "Bilinmiyor":
                o_marka, o_conf = self.validator.validate(crop)
                if o_marka:
                    marka = o_marka
                    conf = o_conf

            # --- 1. KATALOG VERİLERİNİ ÇEK ---
            details = self.get_product_details(marka)
            
            original_code = details.get("original_code", "") if details else ""
            product_name = details.get("product_name", "Bilinmeyen Ürün") if details else "Bilinmeyen Ürün"
            volume = details.get("volume", "") if details else ""
            category = details.get("category", "") if details else ""
            price = details.get("price", "N/A") if details else "N/A" 
            
            # --- 2. RAF VE HİZALAMA HESAPLAMALARI ---
            shelf_pos_x = round((x1 + x2) / (2 * img_w), 2)
            shelf_pos_y = round((y1 + y2) / (2 * img_h), 2)
            is_eye_level = 0.3 <= shelf_pos_y <= 0.6

            # --- 3. JSON OBJESİNE EKLE (Eksikler giderildi!) ---
            image_results["detectedProducts"].append({
                "original_code": original_code,
                "brand": marka,
                "product_name": product_name,
                "volume": volume,
                "category": category,
                "price": price,
                "confidence": round(float(conf), 2),
                "shelfPosition": {"x": shelf_pos_x, "y": shelf_pos_y},
                "isEyeLevel": is_eye_level,
                "bbox": [x1, y1, x2, y2]
            })

            # --- GÖRSEL ETİKETLEME ---
            label = f"{marka} {price}"
            renk = (0, 255, 0) if conf >= self.baraj else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), renk, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), renk, -1) 
            cv2.putText(img, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 1. Analiz Edilmiş Resmi Kaydet
        cikti_resim_yolu = os.path.join(self.output_dir, f"sonuc_{dosya_adi}")
        cv2.imwrite(cikti_resim_yolu, img)

        # 2. JSON Çıktısını Kaydet
        json_yolu = os.path.join(self.output_dir, f"{os.path.splitext(dosya_adi)[0]}.json")
        with open(json_yolu, 'w', encoding='utf-8') as f:
            json.dump(image_results, f, ensure_ascii=False, indent=4)
            
        print(f"✅ Sonuçlar kaydedildi: {json_yolu}")

    def run(self):
        print(f"🚀 İşlem Başlıyor...")
        resim_yollari = glob.glob(os.path.join(self.test_dir, "*.*"))
        resim_yollari = [y for y in resim_yollari if y.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for path in resim_yollari:
            self.process_image(path)
        print(f"\n✨ İşlem Tamamlandı.")