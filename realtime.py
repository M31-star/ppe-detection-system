import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/ppe_model-2/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # ALERT LOGIC
    for r in results:
        classes = r.boxes.cls.tolist()

        if 6 in classes and 0 not in classes:
            print("⚠️ ALERT: Helmet Missing!")

    annotated = results[0].plot()

    cv2.imshow("PPE Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()