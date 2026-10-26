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
USE_PHOTOS = False       # recognise mapped spots from photos (off: navigation uses the step counter)

# ---------- Object detection ----------
# Tried in order. The first one that loads is used ("s" models are far more accurate than "n").
# yolo11s / yolov8s download automatically the first time (needs internet once).
YOLO_MODELS = ["yolo11s.pt", "yolov8s.pt", "yolov8n.pt"]
MIN_CONFIDENCE = 0.40    # ignore guesses YOLO is less sure about (stops "train" / "cat" in a corridor)
CLOSE_HEIGHT = 0.40      # object taller than 40% of the frame = close enough to warn about

# Only things that can actually block a walking person indoors -> what VISTA calls them
OBSTACLES = {
    "person": "person", "chair": "chair", "couch": "sofa", "bench": "bench",
    "dining table": "table", "potted plant": "plant pot", "bicycle": "bicycle",
    "motorcycle": "motorbike", "dog": "dog", "suitcase": "suitcase", "backpack": "bag",
    "handbag": "bag", "bed": "bed", "toilet": "toilet", "refrigerator": "fridge",
    "car": "car", "umbrella": "umbrella",
}
ROOM_EVERY_SECONDS = 2   # read door signs every 2 s

# ---------- Load models ----------
def load_yolo():
    for name in YOLO_MODELS:
        try:
            model = YOLO(name)
            print(f"[YOLO] using {name}")
            return model
        except Exception as e:
            print(f"[YOLO] could not load {name} ({e}) - trying the next one")
    raise RuntimeError("No YOLO model could be loaded")


print("Loading YOLO model...")
yolo_model = load_yolo()

print("Loading MiDaS depth model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

localizer = Localizer("waypoints", start_floor=START_FLOOR) if USE_PHOTOS else None
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

        if localizer:
            waypoint, score = localizer.localize(frame)
            shared["waypoint"] = waypoint
            shared["loc_score"] = score

        if time.time() - last_room_time >= ROOM_EVERY_SECONDS:
            last_room_time = time.time()
            room = room_reader.read(frame)
            shared["room_number"] = room
            floor = floor_from_room(room)
            if floor is not None and localizer:
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
    """The closest obstacle in the walking path: ("chair", "ahead") / ("person", "on your left"), or (None, None)."""
    best = None
    best_height = 0
    for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        name = yolo_model.names[int(cls)]
        if name not in OBSTACLES or float(conf) < MIN_CONFIDENCE:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        centre_x = (x1 + x2) / 2
        height = y2 - y1
        if frame_w * 0.15 < centre_x < frame_w * 0.85 and height > frame_h * CLOSE_HEIGHT:
            if height > best_height:                 # biggest = closest
                best_height = height
                if centre_x < frame_w * 0.4:
                    side = "on your left"
                elif centre_x > frame_w * 0.6:
                    side = "on your right"
                else:
                    side = "ahead"
                best = (OBSTACLES[name], side)
    return best if best else (None, None)

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
prev_hazard = (None, None)
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
    results = yolo_model(frame, conf=MIN_CONFIDENCE, verbose=False)
    annotated = results[0].plot()
    object_names = [yolo_model.names[int(c)] for c in results[0].boxes.cls]
    hazard = detect_hazard(results, frame_w, frame_h)
    is_hazard = hazard[0] is not None and hazard == prev_hazard   # same obstacle 2 frames in a row = real
    prev_hazard = hazard

    # --- Depth / staircase (every DEPTH_EVERY frames) ---
    if frame_i % DEPTH_EVERY == 0:
        depth_display = estimate_depth(frame)
        is_staircase, step_count = detect_staircase(depth_display)

    # --- Latest answers from the background worker ---
    waypoint = shared["waypoint"]
    loc_score = shared["loc_score"]
    room_number = shared["room_number"]
    floor = localizer.current_floor if localizer else START_FLOOR

    # --- Combined output ---
    output = {
        "objects": object_names,
        "staircase_detected": is_staircase,
        "hazard_detected": is_hazard,
        "hazard_object": hazard[0] if is_hazard else None,   # "chair"
        "hazard_side": hazard[1] if is_hazard else None,     # "ahead" / "on your left" / "on your right"
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
              f"obstacle={hazard[0] + ' ' + hazard[1] if is_hazard else '-'} | stairs={is_staircase} | {object_names}")

    # --- Display ---
    label = f"F{floor} {waypoint} ({loc_score})  {fps:.1f}fps  "
    if room_number:
        label += f"ROOM {room_number}  "
    if is_staircase:
        label += f"STAIRCASE ({step_count})  "
    if is_hazard:
        label += f"{hazard[0].upper()} {hazard[1].upper()}"
    cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2)

    cv2.namedWindow("VISTA - Perception", cv2.WINDOW_NORMAL)
    cv2.imshow("VISTA - Perception", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()