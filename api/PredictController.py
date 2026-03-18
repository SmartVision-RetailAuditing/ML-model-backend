from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from api.PredictService import process_image

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = process_image(file.file)
    return JSONResponse(result)