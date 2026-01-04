from pydantic import BaseModel, Field
from typing import List, Dict

# 1. Bounding Box Structure
class BoundingBox(BaseModel):
    x: int = Field(..., description="X coordinate of the top-left corner")
    y: int = Field(..., description="Y coordinate of the top-left corner")
    w: int = Field(..., description="Width of the bounding box")
    h: int = Field(..., description="Height of the bounding box")

# 2. Single Detection Item
class DetectionItem(BaseModel):
    label: str = Field(..., description="Class name of the detected object (e.g., coca_cola)")
    confidence: float = Field(..., description="Confidence score of the detection (between 0.0 and 1.0)")
    bbox: BoundingBox = Field(..., description="Coordinate details of the object on the image")

# 3. Advanced Endpoint Response Schema (/predict/advanced)
class AdvancedResponse(BaseModel):
    timestamp: str = Field(..., description="Timestamp of the processing (ISO format)")
    processing_time_ms: float = Field(..., description="Image processing time in milliseconds")
    total_objects: int = Field(..., description="Total number of objects detected in the image")
    product_counts: Dict[str, int] = Field(..., description="Summary count of each product class (e.g., {'cola': 2, 'chips': 1})")
    detections: List[DetectionItem] = Field(..., description="Detailed list of all detected objects")

# 4. Simple Endpoint Response Schema (/predict/simple)
class SimpleResponse(BaseModel):
    timestamp: str = Field(..., description="Timestamp of the processing")
    total_objects: int = Field(..., description="Total number of objects detected")
    products: Dict[str, int] = Field(..., description="Breakdown of counts per product class")
    saved_image_path: str = Field(..., description="File path or URL of the processed and saved image")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message explaining what went wrong.")