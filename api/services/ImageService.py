import uuid
import os
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

AZURE_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME")

# Blob client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
try:
    container_client.create_container()
except Exception:
    pass  # Zaten varsa geç

def upload_image_to_azure(image_path: str) -> str:
    """Resmi Azure Blob Storage'a yükler ve URL döner"""
    blob_name = f"{uuid.uuid4()}.jpg"
    blob_client = container_client.get_blob_client(blob_name)

    with open(image_path, "rb") as data:
        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type="image/jpeg")
        )
    return blob_client.url