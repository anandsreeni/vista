from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    results = model(frame)
    annotated = results[0].plot()

    # extract plain object names
    names = [model.names[int(c)] for c in results[0].boxes.cls]
    print(names)  # e.g. ["person", "car"]

    cv2.imshow("VISTA - Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()