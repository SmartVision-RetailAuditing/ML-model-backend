from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from api.services.PredictService import process_image

router = APIRouter()

@router.post("/analyze")
async def predict(file: UploadFile = File(...)):
    # process_image içeride file.read() kullandığı için direkt file.file objesini yolluyoruz
    result = process_image(file.file)
    return JSONResponse(content=result)