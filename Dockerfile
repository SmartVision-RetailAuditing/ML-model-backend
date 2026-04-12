# 1. Küçük Python base image
FROM python:3.10-slim

# 2. Çalışma dizini
WORKDIR /app

# 3. Python ayarları
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/.cache

# 4. Sistem bağımlılıkları (minimum)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# 5. requirements
COPY requirements.txt .

# 6. CPU-only PyTorch (CUDA yok)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 7. Diğer bağımlılıklar
RUN pip install --no-cache-dir -r requirements.txt

# 8. Proje dosyaları
COPY . .

# 9. klasörler
RUN mkdir -p ciktilar crops

# 10. port
EXPOSE 8000

# 11. run
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]