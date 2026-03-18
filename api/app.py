from fastapi import FastAPI
from api.PredictController import router

app = FastAPI(title="ML Raf Tanıma API")


app.include_router(router)