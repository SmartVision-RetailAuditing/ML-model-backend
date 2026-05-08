import torch
import os
import logging
from ultralytics import YOLO
import easyocr
from torchvision import transforms

# Gereksiz logları kapat
logging.getLogger('ultralytics').setLevel(logging.ERROR)

class AIModels:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIModels, cls).__new__(cls)
            cls._instance._initialize_models()
        return cls._instance

    def _initialize_models(self):
        print("\n🚀 AI MODELS LOADING... (Singleton Active)")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"💻 Device: {self.device}")

        # =========================
        # YOLO MODEL
        # =========================
        self.det_model = YOLO("weights/best.pt")
        self.det_model.to(self.device)
        print("🎯 YOLO Detection Loaded")

        # =========================
        # DINOv2 MODEL (BASE)
        # =========================
        print("🦖 DINOv2 (Base - 768) yükleniyor...")
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.dinov2_model.eval()
        self.dinov2_model.to(self.device)

        # =========================
        # OCR MODEL (EasyOCR)
        # =========================
        use_gpu = torch.cuda.is_available()
        self.ocr_reader = easyocr.Reader(['tr', 'en'], gpu=use_gpu)
        print("📝 EasyOCR Loaded")

        # =========================
        # DINO PREPROCESS (EKSİK OLAN KISIM EKLENDİ)
        # =========================
        self.dino_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("\n✅ AI MODELS READY")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"OCR GPU Mode: {use_gpu}")
        print(f"Detection Device: {self.det_model.device}")
        print("=" * 50 + "\n")

# GLOBAL INSTANCE
ai_models = AIModels()