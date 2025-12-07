from ultralytics import YOLO


def test_model_loading():
    """Test that a YOLO model loads successfully."""
    model = YOLO("yolo11n.pt")
    assert model is not None
    assert hasattr(model, 'names')  # Check if model has class names
