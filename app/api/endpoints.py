import os
import time
import datetime
import numpy as np
import cv2
from fastapi import APIRouter, File, UploadFile, HTTPException

from ultralytics import YOLO

# Import schemas (ErrorResponse)
from app.Models.schemas import AdvancedResponse, DetectionItem, BoundingBox, SimpleResponse, ErrorResponse

router = APIRouter()

# --- SETUP ---
try:
    model = YOLO("weights/best.pt")
except Exception as e:
    print(f"CRITICAL WARNING: Model failed to load. Details: {e}")
    model = None

RESULT_DIR = "model_results"
os.makedirs(RESULT_DIR, exist_ok=True)


# ==============================================================================
# SCENARIO 1: LEGACY CODE (Visual Output)
# ==============================================================================
@router.post(
    "/predict/simple",
    response_model=SimpleResponse,
    tags=["Simple Workflow"],
    responses={
        200: {"description": "Image processed successfully."},
        400: {"model": ErrorResponse, "description": "Invalid image file (corrupt or unsupported format)."},
        500: {"model": ErrorResponse, "description": "Server error (Model not loaded or file write failed)."}
    }
)
async def predict_simple(file: UploadFile = File(...)):
    """
    Legacy Workflow:
    - Performs real YOLO inference.
    - Uses OpenCV for image loading (matches local training consistency).
    - Saves the processed image to disk.
    - Returns simple counts and the saved file path.
    """
    # 1. Check Model Availability
    if model is None:
        raise HTTPException(status_code=500, detail="Model file is not loaded on the server.")

    try:
        # 2. Read and Validate Image (Using OpenCV)
        image_bytes = await file.read()

        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode image using OpenCV (Loads as BGR, which YOLO prefers)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Check if image was loaded successfully
        if image_np is None:
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

        # 3. Inference
        results = model.predict(source=image_np, conf=0.40)

        counts = {}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(RESULT_DIR, f"{timestamp}.jpg")

        # 4. Processing Results
        for r in results:
            for c in r.boxes.cls:
                label = r.names[int(c)]
                counts[label] = counts.get(label, 0) + 1

            # Plotting (Returns BGR numpy array)
            plotted_img = r.plot()

            # Save using OpenCV (Faster and maintains color consistency)
            cv2.imwrite(output_path, plotted_img)

        return {
            "timestamp": timestamp,
            "total_objects": sum(counts.values()),
            "products": counts,
            "saved_image_path": output_path
        }

    except HTTPException as http_exc:
        # Re-raise known HTTP exceptions
        raise http_exc
    except Exception as e:
        # Catch unexpected server errors
        raise HTTPException(status_code=500, detail=f"Internal Processing Error: {str(e)}")


# ==============================================================================
# SCENARIO 2: NEW STORY (Clean Architecture / JSON Data)
# ==============================================================================
@router.post(
    "/predict/advanced",
    response_model=AdvancedResponse,
    tags=["Advanced Workflow"],
    responses={
        200: {"description": "Successful inference with detailed data."},
        400: {"model": ErrorResponse, "description": "Invalid image file (cannot open or read)."},
        500: {"model": ErrorResponse, "description": "Internal Server Error (Model missing or inference failed)."}
    }
)
async def predict_advanced(file: UploadFile = File(...)):
    """
    Advanced Workflow:
    - Uses OpenCV for robust image loading.
    - Returns detailed Bounding Boxes, Confidence Scores, and Labels.
    - Does NOT save the image to disk (Stateless).
    """
    # 1. Check Model
    if model is None:
        raise HTTPException(status_code=500, detail="Model file is not loaded.")

    start_time = time.time()

    # 2. Read Image (OpenCV Method)
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_np is None:
            raise HTTPException(status_code=400, detail="Invalid image format. Please upload a valid JPG/PNG.")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    try:
        # 3. Inference
        results = model.predict(source=image_np, conf=0.40)

        detections = []
        product_counts = {}

        # 4. Parsing Results
        for r in results:
            for box in r.boxes:
                # Coordinate conversion
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                width = int(x2 - x1)
                height = int(y2 - y1)
                x_coord = int(x1)
                y_coord = int(y1)

                cls_id = int(box.cls[0])
                label_name = r.names[cls_id]
                confidence_score = float(box.conf[0])

                product_counts[label_name] = product_counts.get(label_name, 0) + 1

                detections.append(DetectionItem(
                    label=label_name,
                    confidence=round(confidence_score, 2),
                    bbox=BoundingBox(
                        x=x_coord, y=y_coord, w=width, h=height
                    )
                ))

        process_time = (time.time() - start_time) * 1000

        return AdvancedResponse(
            timestamp=datetime.datetime.now().isoformat(),
            processing_time_ms=round(process_time, 2),
            total_objects=len(detections),
            product_counts=product_counts,
            detections=detections
        )

    except Exception as e:
        # Catch errors during model prediction or logic
        raise HTTPException(status_code=500, detail=f"Model Inference Failed: {str(e)}")