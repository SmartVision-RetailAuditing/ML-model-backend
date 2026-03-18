# 1. Küçük boyutlu Python 3.10 tabanını kullanıyoruz

FROM python:3.10-slim



# 2. Çalışma dizinimizi ayarlıyoruz

WORKDIR /app



# 3. Python'un gereksiz önbellek dosyaları (pyc) oluşturmasını engelliyoruz

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

# DINOv2 modellerinin her seferinde baştan inmemesi için önbellek klasörü

ENV TORCH_HOME=/app/.cache



# 4. İşletim sistemi güncellemelerini yapıp önbelleği siliyoruz (Boyut tasarrufu)

RUN apt-get update && apt-get install -y --no-install-recommends \
&& rm -rf /var/lib/apt/lists/*



# 5. Sadece gereksinimleri kopyalıyoruz (Docker build cache avantajı için)

COPY requirements.txt .



# 6. EN KRİTİK ADIM: Önce sadece CPU destekli PyTorch'u kuruyoruz.

# Bu sayede imaj boyutu ~3GB daha küçük olacak.

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu



# 7. Kalan kütüphaneleri kuruyoruz (Ultralytics içeride torch'u gördüğü için CUDA versiyonunu indirmeyecek)

RUN pip install --no-cache-dir -r requirements.txt

# Sistem paketlerini güncelle ve eksik bağımlılıkları yükle
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*



# 8. Tüm proje dosyalarını (kodlar, modeller, json, pkl) kopyalıyoruz

COPY . .



# 9. Uygulamanın çalışırken hata vermemesi için çıktı klasörlerini oluşturuyoruz

RUN mkdir -p ciktilar crops



# 10. FastAPI'nin çalışacağı portu açıyoruz

EXPOSE 8000


# 11. Uygulamayı başlatıyoruz (Ana dosyanızın adı main.py ise bu şekilde kalabilir)
CMD CD APĞ
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]