import os
import cv2
import re
from collections import Counter, deque

try:
    import pytesseract
except ImportError:
    pytesseract = None

# Default Windows install location; if Tesseract is on PATH this isn't needed
WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Room numbers like 204, 31, 105A. 4+ digit numbers are ignored (usually not room signs)
ROOM_PATTERN = re.compile(r"\b\d{2,3}[A-Za-z]?\b")


class RoomReader:
    def __init__(self, history=5, min_hits=3):
        self.recent = deque(maxlen=history)
        self.min_hits = min_hits         # same number must be read 3 of the last 5 times
        self.enabled = False

        if pytesseract is None:
            print("[ROOM] pytesseract not installed - room numbers OFF (pip install pytesseract)")
            return
        if os.path.exists(WINDOWS_TESSERACT):
            pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT
        try:
            version = pytesseract.get_tesseract_version()
            self.enabled = True
            print(f"[ROOM] Tesseract {version} found - room numbers ON")
        except Exception:
            print("[ROOM] Tesseract program not found - room numbers OFF")

    def read(self, frame):
        if not self.enabled:
            return None

        h, w = frame.shape[:2]
        # Focus on the upper-center area where door signs usually sit
        crop = frame[0:int(h * 0.5), int(w * 0.25):int(w * 0.75)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # small text reads better enlarged
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # auto threshold

        text = pytesseract.image_to_string(thresh, config="--psm 6")
        match = ROOM_PATTERN.search(text)
        found = match.group() if match else None

        self.recent.append(found)
        counts = Counter(x for x in self.recent if x)
        if counts:
            best, hits = counts.most_common(1)[0]
            if hits >= self.min_hits:
                return best
        return None


def floor_from_room(room):
    """'204' -> 2,  '005' -> 0,  '31' -> None (2-digit numbers don't say the floor)"""
    digits = re.sub(r"\D", "", room or "")
    if len(digits) == 3:
        return int(digits[0])
    return None