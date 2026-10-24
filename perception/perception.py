from ultralytics import YOLO
import cv2
import torch
import numpy as np
import json
import os
import time
import threading
from localize import Localizer, pretty
from room_reader import RoomReader, floor_from_room

# ---------- Settings ----------
START_FLOOR = 0          # floor the user starts on (0 = ground floor)
DEPTH_EVERY = 3          # MiDaS on every 3rd frame (stairs don't appear in 0.2 s)
ROOM_EVERY_SECONDS = 2   # read door signs every 2 s

# ---------- Load models ----------
print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")

print("Loading MiDaS depth model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

localizer = Localizer("waypoints", start_floor=START_FLOOR)
room_reader = RoomReader()

# Frames come from the phone via phone_bridge/server.py
FRAME_FILE = "latest_frame.jpg"

# ---------- Shared between main loop and background worker ----------
# Localization + room reading are slow (up to ~1 s), so they run in a
# background thread. The main loop never waits for them - it just uses
# the most recent answer.
shared = {
    "frame": None,
    "waypoint": "unknown",
    "loc_score": 0,
    "room_number": None,
}


def background_worker():
    last_room_time = 0
    while True:
        frame = shared["frame"]
        if frame is None:
            time.sleep(0.1)
            continue

        waypoint, score = localizer.localize(frame)
        shared["waypoint"] = waypoint
        shared["loc_score"] = score

        if time.time() - last_room_time >= ROOM_EVERY_SECONDS:
            last_room_time = time.time()
            room = room_reader.read(frame)
            shared["room_number"] = room
            floor = floor_from_room(room)
            if floor is not None:
                localizer.set_floor(floor, f"(read room {room})")

        time.sleep(0.05)


threading.Thread(target=background_worker, daemon=True).start()

# ---------- Helper functions ----------
def detect_staircase(depth_map):
    h, w = depth_map.shape
    strip = depth_map[h//2:h, w//2-40:w//2+40].astype(np.float32)
    row_means = strip.mean(axis=1).reshape(-1, 1)
    row_means = cv2.GaussianBlur(row_means, (1, 15), 0).flatten()  # smooth noise

    diffs = np.abs(np.diff(row_means))
    threshold = max(diffs.mean() + 2 * diffs.std(), 6)  # absolute floor

    # count distinct jumps, not every row above the threshold
    peaks, prev = 0, False
    for d in diffs:
        cur = d > threshold
        if cur and not prev:
            peaks += 1
        prev = cur

    return bool(peaks >= 4), int(peaks)

def detect_hazard(results, frame_w, frame_h):
    # obstacle = a detected object in the middle of the frame, taller than half the frame
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = [float(v) for v in box]
        centre_x = (x1 + x2) / 2
        if frame_w * 0.3 < centre_x < frame_w * 0.7 and (y2 - y1) > frame_h * 0.5:
            return True
    return False

def estimate_depth(frame):
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
    return cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ---------- Main loop ----------
print("Waiting for frames from the phone...")
last_mtime = 0
frame_i = 0
is_staircase, step_count = False, 0
fps_time = time.time()
fps_count = 0
fps = 0.0

while True:
    # --- Read the latest frame the phone sent ---
    try:
        mtime = os.path.getmtime(FRAME_FILE)
    except FileNotFoundError:
        time.sleep(0.05)
        continue
    if mtime == last_mtime:          # no new frame from phone yet
        time.sleep(0.01)
        continue
    last_mtime = mtime
    frame = cv2.imread(FRAME_FILE)
    if frame is None:                # caught it mid-swap, try next time
        continue
    frame_i += 1
    frame_h, frame_w = frame.shape[:2]
    shared["frame"] = frame          # hand the newest frame to the background worker

    # --- Object detection (every frame) ---
    results = yolo_model(frame, verbose=False)
    annotated = results[0].plot()
    object_names = [yolo_model.names[int(c)] for c in results[0].boxes.cls]
    is_hazard = detect_hazard(results, frame_w, frame_h)

    # --- Depth / staircase (every DEPTH_EVERY frames) ---
    if frame_i % DEPTH_EVERY == 0:
        depth_display = estimate_depth(frame)
        is_staircase, step_count = detect_staircase(depth_display)

    # --- Latest answers from the background worker ---
    waypoint = shared["waypoint"]
    loc_score = shared["loc_score"]
    room_number = shared["room_number"]
    floor = localizer.current_floor

    # --- Combined output ---
    output = {
        "objects": object_names,
        "staircase_detected": is_staircase,
        "hazard_detected": is_hazard,
        "location": pretty(waypoint),    # "Floor 2, computer lab 1" - what gets spoken
        "waypoint": waypoint,            # "f2_computer_lab_1"       - for the route file later
        "floor": floor,
        "room_number": room_number,
        "location_score": int(loc_score),
    }

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

    # --- Status line once a second ---
    fps_count += 1
    if time.time() - fps_time >= 1.0:
        fps = fps_count / (time.time() - fps_time)
        fps_time = time.time()
        fps_count = 0
        print(f"{fps:.1f} fps | floor {floor} | {waypoint} ({loc_score}) | room={room_number} | "
              f"hazard={is_hazard} | stairs={is_staircase} | {object_names}")

    # --- Display ---
    label = f"F{floor} {waypoint} ({loc_score})  {fps:.1f}fps  "
    if room_number:
        label += f"ROOM {room_number}  "
    if is_staircase:
        label += f"STAIRCASE ({step_count})  "
    if is_hazard:
        label += "HAZARD"
    cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2)

    cv2.namedWindow("VISTA - Perception", cv2.WINDOW_NORMAL)
    cv2.imshow("VISTA - Perception", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()