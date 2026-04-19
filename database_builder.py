import os
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ==========================================
# AYARLAR
# ==========================================
REFERENCES_DIR = "references"
OUTPUT_DB = "product_embeddings.pkl"

def main():
    print("🧠 ResNet50 Veritabanı Kurucu Başlatılıyor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan İşlemci: {device}")

    # 1. ResNet50 Modelini Yükle (Sınıflandırma katmanını atıp sadece Özellik Çıkarıcıyı alıyoruz)
    print("Model indiriliyor/yükleniyor (İlk seferde biraz sürebilir)...")
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    
    # Modelin son katmanını (FC) kesip atıyoruz ki bize 2048'lik ham parmak izini versin
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    resnet.to(device)

    # 2. Görüntü Ön İşleme (ResNet50'nin sevdiği standart format)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    veritabani = {}
    basarili = 0

    print("\n📸 Referans Fotoğrafların Parmak İzleri Çıkarılıyor...")
    
    for dosya_adi in os.listdir(REFERENCES_DIR):
        if not dosya_adi.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        dosya_yolu = os.path.join(REFERENCES_DIR, dosya_adi)
        
        try:
            # Resmi aç ve işle
            img = Image.open(dosya_yolu).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            # Parmak izini (Vektör) çıkar
            with torch.no_grad():
                features = resnet(img_tensor)
                features = features.flatten() # 2048'lik düz bir listeye çevir
                # Cosine Similarity için vektörü önceden normalize et (Hız kazandırır)
                features = features / features.norm(p=2) 
                
            # Dosya adını (uzantısız) anahtar olarak kaydet
            urun_adi = os.path.splitext(dosya_adi)[0]
            veritabani[urun_adi] = features.cpu().numpy()
            basarili += 1
            print(f"  -> İşlendi: {urun_adi}")
            
        except Exception as e:
            print(f"HATA ({dosya_adi}): {e}")

    # 3. Veritabanını Kaydet
    with open(OUTPUT_DB, 'wb') as f:
        pickle.dump(veritabani, f)

    print(f"\n✅ İŞLEM TAMAM! {basarili} ürünün parmak izi '{OUTPUT_DB}' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()