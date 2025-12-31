import datetime
from ultralytics import YOLO

# 1. Load the Local Model
model = YOLO("weights/best.pt")

# 2. Set Timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 3. Define Image Path
image_path = "images/test1.jpeg"

try:
    # 4. Perform Prediction
    results = model.predict(source=image_path, conf=0.40)

    print("\n" + "=" * 35)
    print(f"📦 FAST ANALYSIS REPORT | {timestamp}")
    print("=" * 35)

    for r in results:
        # Count total objects found
        total_objects = len(r.boxes)
        print(f"Total Objects Found: {total_objects}")
        print("-" * 35)

        # Count individual items
        counts = {}
        for c in r.boxes.cls:
            label = r.names[int(c)]
            counts[label] = counts.get(label, 0) + 1

        # Print counting results
        if not counts:
            print("❌ No products found. Please check lighting or confidence threshold.")
        else:
            for item, count in counts.items():
                print(f"✅ {item}: {count} pcs")

        # Save the visual result
        output_name = f"model_results/result_{timestamp}.jpg"
        r.save(filename=output_name)

    print("-" * 35)
    print(f"📁 Image saved: {output_name}")
    print("=" * 35)

except Exception as e:
    print(f"⚠️ An error occurred: {e}")