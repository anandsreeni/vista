from ultralytics import YOLO
import cv2
import torch
import numpy as np
import json

# ---------- Load models ----------
print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")

print("Loading MiDaS depth model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

cam = cv2.VideoCapture(0)

# ---------- Helper functions ----------
def detect_staircase(depth_map):
    h, w = depth_map.shape
    strip = depth_map[h//2:h, w//2-40:w//2+40]
    row_means = strip.mean(axis=1)
    diffs = np.abs(np.diff(row_means))
    threshold = diffs.mean() + diffs.std()
    step_count = np.sum(diffs > threshold)
    return bool(step_count >= 4), int(step_count)

def detect_hazard(depth_map):
    h, w = depth_map.shape
    center = depth_map[h//2:h, w//2-50:w//2+50]
    max_diff = np.max(center) - np.min(center)
    threshold = 60
    return bool(max_diff > threshold)

# ---------- Main loop ----------
while True:
    ret, frame = cam.read()
    if not ret:
        break

    # --- Object detection ---
    results = yolo_model(frame)
    annotated = results[0].plot()
    object_names = [yolo_model.names[int(c)] for c in results[0].boxes.cls]

    # --- Depth estimation ---
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    is_staircase, step_count = detect_staircase(depth_display)
    is_hazard = detect_hazard(depth_display)

    # --- Combined output (this is what Person 2 will consume) ---
    output = {
        "objects": object_names,
        "staircase_detected": is_staircase,
        "hazard_detected": is_hazard,
    }
    print(output)

    # Optional: save latest output to a shared file Person 2 can read from
    with open("latest_perception.json", "w") as f:
        json.dump(output, f)

    # --- Display ---
    label = ""
    if is_staircase:
        label += f"STAIRCASE ({step_count})  "
    if is_hazard:
        label += "HAZARD"
    if label:
        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    cv2.imshow("VISTA - Perception", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()