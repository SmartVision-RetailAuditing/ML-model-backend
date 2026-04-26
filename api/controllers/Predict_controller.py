# controllers/Predict_controller.py
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from services.ImageService import ImageService
from services.PredictService import PredictService
import os
import time
import uuid
from core import config

router = APIRouter()
predict_service = PredictService(catalog_path="product_catalog_sut.json")

os.makedirs("output_images", exist_ok=True)


@router.post("/analyze")
async def predict_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    total_start_time = time.time()

    image_bytes = await file.read()
    img = ImageService.bytes_to_cv2(image_bytes)

    if img is None:
        return {"error": "Geçersiz veya bozuk resim dosyası."}

    ai_start_time = time.time()

    # 1. Analiz
    analysis_result = predict_service.process_image(img)
    detected_products = analysis_result["products"]
    detected_tags = analysis_result["tags"]

    ai_end_time = time.time()
    ai_duration = round(ai_end_time - ai_start_time, 2)

    # 2. Çizim ve Kaydetme
    output_path = os.path.join("output_images", f"sonuc_{file.filename}")
    ImageService.draw_boxes_and_save(img, detected_products, detected_tags, output_path)

    # --- YENİ MANTIK: AZURE YÜKLEMESİNİ ARKA PLANA AT ---
    if config.USE_AZURE and config.AZURE_BLOB_CONNECTION_STRING:
        # Dosya adını ve URL'yi şimdiden (yüklemeden önce) belirliyoruz
        blob_name = f"{uuid.uuid4()}.jpg"
        account_name = config.AZURE_BLOB_CONNECTION_STRING.split("AccountName=")[1].split(";")[0]
        azure_url = f"https://{account_name}.blob.core.windows.net/{config.AZURE_BLOB_CONTAINER_NAME}/{blob_name}"

        # Asıl yükleme işini arka plana postalıyoruz (API'yi bekletmez!)
        background_tasks.add_task(ImageService.upload_to_azure_bg, output_path, blob_name)
    else:
        azure_url = None
    # -----------------------------------------------------

    total_end_time = time.time()
    total_duration = round(total_end_time - total_start_time, 2)

    print(f"⏱️ Yapay Zeka Analizi: {ai_duration} saniye")
    print(f"⏱️ Toplam İşlem (Azure Yüklemesi Arkada Yapılıyor): {total_duration} saniye")
    print("-" * 40)

    return {
        "azure_blob_url": azure_url,
        "products": detected_products,
        "processing_time_seconds": total_duration
    }