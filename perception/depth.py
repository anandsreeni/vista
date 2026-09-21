import cv2
import torch
import numpy as np

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

cam = cv2.VideoCapture(0)

def detect_staircase(depth_map):
    """
    Looks at a vertical strip down the center-bottom of the frame.
    A staircase shows up as a repeating pattern of sudden depth jumps
    (each step is a bit closer/higher than the one before it).
    """
    h, w = depth_map.shape
    strip = depth_map[h//2:h, w//2-40:w//2+40]  # center-bottom strip
    row_means = strip.mean(axis=1)  # average depth per row in the strip

    # Look at how much depth changes between consecutive rows
    diffs = np.abs(np.diff(row_means))
    threshold = diffs.mean() + diffs.std()  # adaptive threshold
    step_count = np.sum(diffs > threshold)

    # If there are several sharp transitions in that strip, likely a staircase
    return step_count >= 4, step_count

def detect_hazard(depth_map):
    """
    Flags a hazard if there's a very sudden, large depth jump
    right in front of the camera (possible hole/step/drop-off).
    """
    h, w = depth_map.shape
    center = depth_map[h//2:h, w//2-50:w//2+50]
    max_diff = np.max(center) - np.min(center)
    threshold = 60  # tune this after testing on your camera
    return max_diff > threshold

while True:
    ret, frame = cam.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    is_staircase, step_count = detect_staircase(depth_display)
    is_hazard = detect_hazard(depth_display)
    print(f"step_count={step_count}")

    label = ""
    if is_staircase:
        label += f"STAIRCASE LIKELY ({step_count} steps)  "
    if is_hazard:
        label += "HAZARD AHEAD"

    if label:
        print(label)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    cv2.imshow("Depth Map", depth_display)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()