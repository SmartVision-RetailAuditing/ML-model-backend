# core/config.py
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# AZURE AYARLARI
# Kendi GPU sunucuna geçtiğin için varsayılanı False yapmak veya
# .NET backend'in hala Azure Blob Storage kullanıyorsa aktif bırakmak sana kalmış.
USE_AZURE = os.getenv("USE_AZURE", "False").lower() == "true"
AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME")

# AI VE MODEL AYARLARI
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.45"))  # Det_model (Kutu bulucu) için hala geçerli
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.50"))

# DINOv2 Kosinüs Benzerliği Eşiği (0.80 - 0.85 idealdir)
DINO_SIMILARITY_THRESHOLD = float(os.getenv("DINO_SIMILARITY_THRESHOLD", "0.50"))

EDGE_TOLERANCE = int(os.getenv("EDGE_TOLERANCE", "2"))

try:
    SHELF_THRESHOLDS = [int(x) for x in os.getenv("SHELF_THRESHOLDS", "400,800,1200,1600").split(",")]
except ValueError:
    SHELF_THRESHOLDS = [400, 800, 1200, 1600]