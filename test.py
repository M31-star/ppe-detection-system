from ultralytics import YOLO
import os
import cv2

# -------------------------------
# 1. Load trained model
# -------------------------------
model = YOLO("runs/detect/ppe_model-2/weights/best.pt")

# -------------------------------
# 2. Define paths
# -------------------------------
input_folder = "dataset/images/test"
output_folder = "output"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# 3. Process all images
# -------------------------------
image_count = 0

for image_name in os.listdir(input_folder):

    image_path = os.path.join(input_folder, image_name)

    # Skip non-image files
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Skipped: {image_name}")
        continue

    # Run detection
    results = model(img)

    # Draw bounding boxes
    annotated = results[0].plot()

    # Save output image
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, annotated)

    print(f"✅ Processed: {image_name}")
    image_count += 1

# -------------------------------
# 4. Final message
# -------------------------------
print("\n🎯 DONE")
print(f"Total images processed: {image_count}")
print("Results saved in 'output' folder")