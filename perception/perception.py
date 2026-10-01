from ultralytics import YOLO
import cv2
import torch
import numpy as np
import json
import os
import time
from scene_classifier import Localizer
from room_reader import RoomReader

# ---------- Load models ----------
print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")

print("Loading MiDaS depth model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# ---------- Localizer (CLIP, no reference photos needed) ----------
print("Loading CLIP localizer...")
localizer = Localizer()
frame_i = 0
location, loc_score = "unknown", 0

# ---------- Room number reader (OCR) ----------
room_reader = RoomReader()
room_number = None

cam = cv2.VideoCapture(0)

# ---------- Helper functions ----------
def detect_staircase(depth_map):
    h, w = depth_map.shape
    strip = depth_map[h//2:h, w//2-40:w//2+40].astype(np.float32)
    row_means = strip.mean(axis=1).reshape(-1, 1)
    row_means = cv2.GaussianBlur(row_means, (1, 15), 0).flatten()  # smooth noise

    diffs = np.abs(np.diff(row_means))
    threshold = max(diffs.mean() + 2 * diffs.std(), 6)  # absolute floor

    peaks, prev = 0, False
    for d in diffs:
        cur = d > threshold
        if cur and not prev:
            peaks += 1
        prev = cur

    return bool(peaks >= 4), int(peaks)

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

    frame_i += 1

    # --- Localization (every 15th frame; CLIP is heavier than ORB) ---
    if frame_i % 15 == 0:
        location, loc_score = localizer.localize(frame)

    # --- Room number OCR (every 10th frame) ---
    if frame_i % 10 == 0:
        room_number = room_reader.read(frame)

    # --- Combined output ---
    output = {
        "objects": object_names,
        "staircase_detected": is_staircase,
        "hazard_detected": is_hazard,
        "location": location,
        "location_score": int(loc_score),
        "room_number": room_number,
    }
    print(output)

    with open("latest_perception.tmp", "w") as f:
        json.dump(output, f)

    for attempt in range(5):
        try:
            os.replace("latest_perception.tmp", "latest_perception.json")
            break
        except PermissionError:
            time.sleep(0.05)
    else:
        print("Warning: could not update latest_perception.json this frame")

    # --- Display ---
    label = f"{location} ({loc_score}%)  "
    if room_number:
        label += f"ROOM {room_number}  "
    if is_staircase:
        label += f"STAIRCASE ({step_count})  "
    if is_hazard:
        label += "HAZARD"
    cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)

    cv2.namedWindow("VISTA - Perception", cv2.WINDOW_NORMAL)
    cv2.imshow("VISTA - Perception", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()