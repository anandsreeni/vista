import os
import re
import cv2
from collections import Counter, deque

FLOOR_PREFIX = re.compile(r"^f(\d+)_")
CONNECTOR_WORDS = ("stairs", "lift", "elevator")   # waypoints that lead to other floors


def floor_of(name):
    """'f2_room_204' -> 2,  'entrance' -> None"""
    m = FLOOR_PREFIX.match(name)
    if m:
        return int(m.group(1))
    return None


def pretty(name):
    """'f2_computer_lab_1' -> 'Floor 2, computer lab 1',  'f0_entrance' -> 'Ground floor, entrance'"""
    if name == "unknown":
        return "unknown"
    floor = floor_of(name)
    place = FLOOR_PREFIX.sub("", name).replace("_", " ")
    if floor is None:
        return place
    if floor == 0:
        return f"Ground floor, {place}"
    return f"Floor {floor}, {place}"


class Localizer:
    """
    Pre-mapped, floor-aware localization with ORB feature matching.

    Photos saved by Map mode, names start with the floor:
        waypoints/
            f0_entrance/   f0_stairs/   f2_room_204/   f2_stairs/ ...

    Only waypoints on the CURRENT floor are checked, plus the stairs/lift
    waypoints one floor above and below. Recognising those is how the
    floor changes. This keeps matching fast and stops floor 2 being
    confused with floor 3.
    """

    def __init__(self, waypoint_dir="waypoints", start_floor=0,
                 min_matches=15, margin=1.4, history=3):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.min_matches = min_matches   # below this = weak match = unknown
        self.margin = margin             # best must beat 2nd best by 40%
        self.recent = deque(maxlen=history)
        self.current_floor = start_floor
        self.refs = {}                   # name -> [descriptors of photo 1, photo 2, ...]

        print("\n========== LOADING WAYPOINTS ==========")
        if not os.path.isdir(waypoint_dir):
            print(f"[ERROR] Folder not found: {waypoint_dir} - run Map mode first")
        else:
            for name in sorted(os.listdir(waypoint_dir)):
                folder = os.path.join(waypoint_dir, name)
                if not os.path.isdir(folder):
                    continue
                descriptor_list = []
                for file in sorted(os.listdir(folder)):
                    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    kp, des = self.orb.detectAndCompute(img, None)
                    if des is None or len(des) < 2:
                        print(f"[SKIP] {name}/{file} - no features (blank wall?)")
                        continue
                    descriptor_list.append(des)
                if not descriptor_list:
                    print(f"[ERROR] {name}: no usable photos")
                    continue
                self.refs[name] = descriptor_list
                warning = "" if floor_of(name) is not None else "   <-- no f<number>_ prefix, checked on every floor"
                print(f"[OK] {name}: {len(descriptor_list)} photos{warning}")
        print(f"Starting on floor {self.current_floor}")
        print("========================================\n")

    def candidates(self):
        """Waypoints worth checking right now."""
        result = []
        for name in self.refs:
            floor = floor_of(name)
            if floor is None or floor == self.current_floor:
                result.append(name)
            elif abs(floor - self.current_floor) == 1 and any(w in name for w in CONNECTOR_WORDS):
                result.append(name)
        return result

    def set_floor(self, floor, reason=""):
        if floor != self.current_floor:
            print(f"[FLOOR] {self.current_floor} -> {floor} {reason}")
            self.current_floor = floor
            self.recent.clear()          # old floor's guesses no longer count

    def score(self, frame_des, ref_des):
        matches = self.bf.knnMatch(frame_des, ref_des, k=2)
        good = 0
        for pair in matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.65 * n.distance:    # Lowe's ratio test
                good += 1
        return good

    def raw_localize(self, frame):
        """Best guess for this single frame, no smoothing."""
        names = self.candidates()
        if frame is None or not names:
            return "unknown", 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(des) < 2:
            return "unknown", 0

        scores = {}
        for name in names:
            scores[name] = max(self.score(des, ref_des) for ref_des in self.refs[name])

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_name, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0

        if best_score < self.min_matches:
            return "unknown", best_score                 # weak match
        if best_score < second_score * self.margin:
            return "unknown", best_score                 # two places look alike
        return best_name, best_score

    def localize(self, frame):
        """Smoothed: a waypoint counts only if it won 2 of the last 3 checks."""
        name, score = self.raw_localize(frame)
        self.recent.append(name)
        best, hits = Counter(self.recent).most_common(1)[0]
        if hits < 2 or best == "unknown":
            return "unknown", score

        floor = floor_of(best)
        if floor is not None:
            self.set_floor(floor, f"(recognised {best})")
        return best, score