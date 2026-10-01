import cv2
import re
import pytesseract
from collections import Counter, deque

# Update this path if your Tesseract install location is different
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ROOM_PATTERN = re.compile(r"\b\d{1,4}[A-Za-z]?\b")

class RoomReader:
    def __init__(self, history=5, min_hits=3):
        self.recent = deque(maxlen=history)
        self.min_hits = min_hits

    def read(self, frame):
        h, w = frame.shape[:2]
        # Focus on the upper-center area where door signs usually sit
        crop = frame[0:int(h * 0.5), int(w * 0.25):int(w * 0.75)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        text = pytesseract.image_to_string(thresh, config="--psm 7")
        match = ROOM_PATTERN.search(text)
        found = match.group() if match else None

        self.recent.append(found)
        counts = Counter(x for x in self.recent if x)
        if counts:
            best, hits = counts.most_common(1)[0]
            if hits >= self.min_hits:
                return best
        return None