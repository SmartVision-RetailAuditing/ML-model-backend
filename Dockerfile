<<<<<<< HEAD
# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build-time system deps (gcc etc. for compiling some pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/.cache

# Only the runtime libs your app actually needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages \
                    /usr/local/lib/python3.10/site-packages

# Copy executables (uvicorn, etc.)
COPY --from=builder /usr/local/bin \
                    /usr/local/bin

# Copy app source
COPY . .

RUN mkdir -p ciktilar crops

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
=======
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
>>>>>>> origin/development
