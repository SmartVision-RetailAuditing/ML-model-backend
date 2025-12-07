from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n.pt")

# Define source as YouTube video URL
source = "https://www.youtube.com/shorts/iRdXmjIgYQg"
# Define path to the image file
# source = "path/to/image.jpg"
# Define path to directory containing images and videos for inference
# source = "path/to/dir"

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
