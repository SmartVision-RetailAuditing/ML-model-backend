# SmartVision AI Backend (YOLOv11)

This repository hosts the **computer vision API** for the Smart Vision project. It uses **YOLOv11** wrapped in **FastAPI** to detect, classify, and count retail products on store shelves in real-time.

## 📂 Project Structure

```text
smart-vision/
├── app/                 # FastAPI Source Code
│   ├── main.py          # Entry Point
│   ├── api/             # Endpoints
│   └── models/          # Pydantic Schemas
├── weights/             # YOLO Model Weights (best.pt)
├── model_results/       # Processed Images Output
├── train.py             # Model Training Script
├── requirements.txt     # Dependencies
└── README.md
🛠 Installation
1. Clone the Repository

Bash

git clone [https://github.com/SmartVision-RetailAuditing/ML-model-backend.git](https://github.com/SmartVision-RetailAuditing/ML-model-backend.git)
cd smart-vision-yolo
2. Set Up Virtual Environment

Windows:

PowerShell

python -m venv venv
.\venv\Scripts\activate
macOS / Linux:

Bash

python3 -m venv venv
source venv/bin/activate
3. Install Dependencies

Bash

pip install -r requirements.txt
(Ensure weights/best.pt exists in the root directory before running.)

🚀 Running the API
Start the local server with hot-reloading:

Bash

uvicorn app.main:app --reload
Swagger Documentation: http://127.0.0.1:8000/docs

Endpoints:

POST /predict/simple: Returns product counts + Image URL (For Mobile).

POST /predict/advanced: Returns detailed Bounding Boxes + Confidence JSON (For Backend).