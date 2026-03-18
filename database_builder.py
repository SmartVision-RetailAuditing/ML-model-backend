import os
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image

# ==========================================
# AYARLAR
# ==========================================
REFERENCES_DIR = "references"
OUTPUT_DB = "product_embeddings.pkl"


def main():
    print("🧠 DINOv2 Veritabanı Kurucu Başlatılıyor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan İşlemci: {device}")

    # 1. DINOv2 Modelini Yükle (ResNet50 yerine)
    print("DINOv2 modeli indiriliyor/yükleniyor...")
    # 'dinov2_vits14' küçük ve hızlıdır. Daha yüksek doğruluk için 'dinov2_vitb14' kullanabilirsiniz.
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2.eval()
    dinov2.to(device)

    # 2. Görüntü Ön İşleme (DINOv2 için standart ayarlar)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # 224, 14'e tam bölünür (DINOv2 patch boyutu)
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
            img = Image.open(dosya_yolu).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            # DINOv2 ile parmak izini çıkar
            with torch.no_grad():
                # DINOv2 doğrudan özellikleri döndürür, ResNet gibi katman kesmeye gerek yoktur.
                features = dinov2(img_tensor)
                features = features.flatten()
                features = features / features.norm(p=2)  # Vektörü normalize et

            urun_adi = os.path.splitext(dosya_adi)[0]
            veritabani[urun_adi] = features.cpu().numpy()
            basarili += 1
            print(f"  -> İşlendi: {urun_adi}")

        except Exception as e:
            print(f"HATA ({dosya_adi}): {e}")

    # 3. Veritabanını Kaydet
    with open(OUTPUT_DB, 'wb') as f:
        pickle.dump(veritabani, f)

    print(f"\n✅ İŞLEM TAMAM! {basarili} ürünün DINOv2 parmak izi kaydedildi.")


if __name__ == "__main__":
    main()