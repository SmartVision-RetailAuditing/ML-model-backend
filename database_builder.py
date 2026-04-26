import os
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image

# ==========================================
# AYARLAR
# ==========================================
REFERENCES_DIR = "references"
# Base model için isimlendirmeyi yaptık
OUTPUT_DB = "weights/dinov2_large_product_embeddings.pkl"

def main():
    print("🧠 DINOv2 (BASE) Vektör Veritabanı Kurucu Başlatılıyor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan İşlemci: {device}")

    # 1. DINOv2 BASE Modelini Yükle (Altın Oran: vitb14)
    print("DINOv2 ViT-Base modeli indiriliyor/yükleniyor...")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dinov2_model.eval()
    dinov2_model.to(device)

    # 2. Ön İşleme (224x224 Base model için de en ideal hız/performans oranıdır)
    preprocess = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    veritabani = {}
    basarili = 0

    # BASE model 768 boyutlu vektör çıkarır!
    print("\n📸 Referans Fotoğrafların 768 Boyutlu Parmak İzleri Çıkarılıyor...")

    for dosya_adi in os.listdir(REFERENCES_DIR):
        # .webp desteği
        if not dosya_adi.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue

        dosya_yolu = os.path.join(REFERENCES_DIR, dosya_adi)

        try:
            # WebP şeffaflık sorununu çözmek için convert('RGB')
            img = Image.open(dosya_yolu).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = dinov2_model(img_tensor)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                features = features.squeeze(0).cpu().numpy()

            urun_adi = os.path.splitext(dosya_adi)[0]
            veritabani[urun_adi] = features
            basarili += 1
            print(f"  -> İşlendi: {urun_adi} (Vektör Boyutu: {features.shape[0]})")

        except Exception as e:
            print(f"HATA ({dosya_adi}): {e}")

    # Klasör yoksa oluştur
    os.makedirs(os.path.dirname(OUTPUT_DB) or '.', exist_ok=True)

    with open(OUTPUT_DB, 'wb') as f:
        pickle.dump(veritabani, f)

    print(f"\n✅ İŞLEM TAMAM! {basarili} ürünün parmak izi '{OUTPUT_DB}' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()