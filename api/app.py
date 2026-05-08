# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controllers.Predict_controller import router as predict_router

app = FastAPI(title="Market Raf Analiz API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, tags=["Prediction"])

@app.get("/")
def root():
    return {"mesaj": "Sistem aktif. Backend'den /api/v1/predict adresine istek atabilirsin."}