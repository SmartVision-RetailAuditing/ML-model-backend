from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from api.services.PredictService import process_image

router = APIRouter()

@router.post("/analyze")
async def predict(file: UploadFile = File(...)):
    result = process_image(file.file)
    return JSONResponse(result)