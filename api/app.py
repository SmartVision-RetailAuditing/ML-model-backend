from fastapi import FastAPI
<<<<<<< HEAD
from api.controllers.PredictController import router

app = FastAPI(
    title="Smart Vision - Raf Tanıma API",
    description="YOLO ve OCR tabanlı perakende raf analiz sistemi",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
=======
from fastapi.middleware.cors import CORSMiddleware
from api.controllers.PredictController import router as predict_router

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
    return {"mesaj": "Sistem aktif"}
>>>>>>> origin/development
