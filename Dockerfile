# 1. Python 3.10 Slim
FROM python:3.10-slim

# 2. Konteyner içindeki ANA çalışma dizinimiz
WORKDIR /app

# 3. OpenCV ve sistem kütüphaneleri
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. requirements'ı kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Tüm projeyi kopyala (api klasörü dahil her şey /app içine gelir)
COPY . .

# --- KRİTİK EKLENTİ ---
# Kodlarımız, YOLO ağırlıklarımız ve JSON kataloğumuz 'api' klasörünün içinde.
# Uygulamayı başlatmadan önce o klasörün içine giriyoruz ki Python dosyaları bulabilsin.
WORKDIR /app/api
# ----------------------

# 6. Portu aç
EXPOSE 8000

# 7. Uygulamayı başlat (Artık api klasörünün içinde olduğu için 'app:app' tıkır tıkır çalışır)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]