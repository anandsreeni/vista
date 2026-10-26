"""
VISTA voice guide - turns "take me to the office" into step-by-step spoken directions.

How it knows where the user is:
  1. Step counter + turn sensor from the phone   (main guide between spots)
  2. "I'm at the entrance" said by the user      (sets the starting point)
  3. Camera recognising a mapped spot            (starting point, or fallback when the step counter is off)

Directions come from route.json.
"""
import json
import os
import re
import time
import heapq
import logging

# ---------- Files (paths work no matter which folder you run this from) ----------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PERCEPTION_FILE = os.path.join(ROOT, "perception", "latest_perception.json")
SCENE_GRAPH_FILE = os.path.join(ROOT, "scene_graph", "scene_graph.json")
ROUTE_FILE = os.path.join(HERE, "route.json")
DESTINATION_FILE = os.path.join(HERE, "destination.json")
SPEECH_FILE = os.path.join(HERE, "latest_speech.json")
HERE_FILE = os.path.join(HERE, "here.json")                # "I'm at the entrance" from the phone
MOTION_FILE = os.path.join(HERE, "motion.json")            # step count + turn from the phone
ASK_FILE = os.path.join(HERE, "ask.json")                  # "where am I?" from the phone

# ---------- Timing ----------
HAZARD_REPEAT = 6        # seconds before the same safety warning is repeated
STAIRS_REPEAT = 8
NAV_REPEAT = 30          # repeat the current direction if the user is standing still
SAFETY_GAP = 2.5         # let a warning finish before a direction is spoken
GEMINI_EVERY = 20        # scene description only when NO destination is set

# ---------- Step counting ----------
TURN_DONE_DEG = 60       # turned at least this much = the turn is done
TURN_AROUND_DEG = 120    # for "Turn around"
NEAR_STEPS = 3           # "In about 3 steps you will reach junction 1"
OVERSHOOT_STEPS = 8      # this many steps past the turn = probably missed it

# ---------- Gemini (optional: only used when no destination is chosen) ----------
USE_GEMINI = True
client = None
API_KEY = os.environ.get("GEMINI_API_KEY")
if USE_GEMINI and API_KEY:
    from google import genai
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    client = genai.Client(api_key=API_KEY)
else:
    print("Gemini OFF (no GEMINI_API_KEY) - navigation and warnings still work")

MODEL_NAME = "gemini-flash-lite-latest"
FALLBACK_MODEL = "gemini-2.5-flash-lite"

SYSTEM_INSTRUCTION = """You are VISTA, a calm walking guide for a visually impaired user.
You will be given the user's current location and the most recent landmark the camera noticed.
Respond with ONE short spoken sentence guiding or informing them — like a real-time narrator, not a report.
Never invent details that weren't given to you."""


# ---------- Small helpers ----------
def read_json(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return default


def speak(text):
    print(f"VISTA (to phone): {text}")
    message = {"id": int(time.time() * 1000), "text": text}
    tmp = SPEECH_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(message, f)
    for attempt in range(5):
        try:
            os.replace(tmp, SPEECH_FILE)
            return
        except PermissionError:          # Windows: server.py is reading it
            time.sleep(0.02)


def fresh_motion():
    """Latest step count + turn from the phone, or None if the sensor isn't sending."""
    m = read_json(MOTION_FILE, None)
    if m and time.time() - m.get("t", 0) < 3:
        return m
    return None


# ---------- Route ----------
def place_of(waypoint):
    """Same place seen from the other direction: 'f0_staffroom__back' -> 'f0_staffroom'"""
    return waypoint.split("__")[0]


def load_route():
    route = read_json(ROUTE_FILE, {"names": {}, "edges": []})
    edges = {}                           # "f0_entrance" -> {"f0_junction_1": "Walk straight..."}
    for e in route.get("edges", []):
        edges.setdefault(e["from"], {})[e["to"]] = e["say"]

    # Facing the other way at the same place = "turn around" (added automatically)
    all_spots = set(edges)
    for targets in list(edges.values()):
        all_spots.update(targets)
    for a in all_spots:
        for b in all_spots:
            if a != b and place_of(a) == place_of(b):
                edges.setdefault(a, {}).setdefault(b, "Turn around.")
    start = route.get("start", "f0_entrance")      # where a trip starts if we don't know where the user is
    return route.get("names", {}), edges, start


def spoken_name(waypoint, names):
    waypoint = place_of(waypoint)
    if waypoint in names:
        return names[waypoint]
    place = waypoint.split("_", 1)[1] if waypoint[:1] == "f" and waypoint[1:2].isdigit() else waypoint
    return place.replace("_", " ")       # "f0_junction_1" -> "junction 1"


def walk_length(say):
    """Steps in a direction: 'Walk straight about 13 steps...' -> 13. 'Turn around.' -> 2."""
    numbers = re.findall(r"(\d+)\s*steps", say)
    if numbers:
        return sum(int(n) for n in numbers)
    if say.strip().lower().startswith("turn around"):
        return 2
    return 10                            # no step count written: assume a short walk


def find_path(start, goal, edges):
    """Dijkstra: the route with the FEWEST STEPS from start to goal, or None."""
    queue = [(0, [start])]
    best = {start: 0}
    while queue:
        steps, path = heapq.heappop(queue)
        here = path[-1]
        if place_of(here) == goal:
            return path
        if steps > best.get(here, float("inf")):
            continue
        for nxt, say in edges.get(here, {}).items():
            total = steps + walk_length(say)
            if total < best.get(nxt, float("inf")):
                best[nxt] = total
                heapq.heappush(queue, (total, path + [nxt]))
    return None


def path_steps(path, edges):
    return sum(walk_length(edges[a][b]) for a, b in zip(path, path[1:]))


# ---------- Following one walk with the step counter ----------
TURN_RE = re.compile(r"then turn (left|right)", re.I)
SIDE_RE = re.compile(r"on your (left|right)", re.I)


class Segment:
    """One walk between two spots, e.g. entrance -> junction 1 (13 steps, then turn left)."""

    def __init__(self, frm, to, say, to_name, motion):
        self.frm = frm
        self.to = to
        self.to_name = to_name
        self.steps_needed = walk_length(say)
        turn = TURN_RE.search(say)
        self.end_turn = turn.group(1).lower() if turn else None          # turn at the END of the walk
        self.turn_around = say.strip().lower().startswith("turn around")
        side = SIDE_RE.search(say)
        self.side = side.group(1).lower() if side else None              # door on your left/right
        self.steps0 = motion["steps"] if motion else None
        self.yaw0 = motion["yaw"] if motion else None
        self.near_said = False
        self.turn_said = False
        self.wrong_said = False
        self.overshoot_said = False

    def walked(self, motion):
        if self.steps0 is None or motion is None:
            return 0
        return motion["steps"] - self.steps0

    def update(self, motion):
        """Returns (sentence to say or None, spot reached or None, reached by turning?)."""
        if self.steps0 is None:                                         # first motion reading
            self.steps0 = motion["steps"]
            self.yaw0 = motion["yaw"]
            return None, None, False
        walked = motion["steps"] - self.steps0
        turned = motion["yaw"] - self.yaw0                              # + = left, - = right
        n = self.steps_needed

        if self.turn_around:
            if abs(turned) >= TURN_AROUND_DEG:
                return None, self.to, True
            return None, None, False

        if self.end_turn:
            want = 1 if self.end_turn == "left" else -1
            if walked >= n * 0.6 and turned * want >= TURN_DONE_DEG:
                return None, self.to, True                              # turned the right way
            if abs(turned) < 30:
                self.wrong_said = False
            if walked >= n * 0.6 and turned * want <= -TURN_DONE_DEG and not self.wrong_said:
                self.wrong_said = True
                return f"That is the wrong way. Turn back, then turn {self.end_turn}.", None, False
            if not self.near_said and n >= 6 and walked >= n - NEAR_STEPS:
                self.near_said = True
                return (f"In about {NEAR_STEPS} steps you will reach {self.to_name}. "
                        f"Turn {self.end_turn} there."), None, False
            if not self.turn_said and walked >= n:
                self.turn_said = True
                return f"You are at {self.to_name}. Turn {self.end_turn} now.", None, False
            if not self.overshoot_said and walked >= n + OVERSHOOT_STEPS:
                self.overshoot_said = True
                return "You may have passed the turn. Stop, turn around, and walk back a few steps.", None, False
            return None, None, False

        # plain walk to a door / spot - arrives when the step count is reached
        if not self.near_said and n >= 6 and walked >= n - NEAR_STEPS:
            self.near_said = True
            return f"About {NEAR_STEPS} more steps to {self.to_name}.", None, False
        if walked >= n:
            return None, self.to, False
        return None, None, False


# ---------- Gemini scene description (kept from the original version) ----------
def ask_gemini(prompt_text, max_retries=3):
    for model_name in [MODEL_NAME, FALLBACK_MODEL]:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[SYSTEM_INSTRUCTION, prompt_text],
                )
                return response.text.strip()
            except Exception as e:
                msg = str(e)
                if "429" in msg or "404" in msg:
                    break
                if "503" in msg and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                elif "503" in msg:
                    break
                else:
                    raise
    raise RuntimeError("All Gemini models unavailable right now.")


def gemini_scene_sentence():
    graph = read_json(SCENE_GRAPH_FILE, {"nodes": [], "current_location": "unknown"})
    nodes = graph.get("nodes", [])
    landmark = max(nodes, key=lambda n: n.get("last_seen", "")) if nodes else None
    landmark_text = f"{landmark['name']}" if landmark else "nothing notable yet"
    prompt = f"Current location: {graph.get('current_location', 'unknown')}\nMost recent landmark: {landmark_text}"
    return ask_gemini(prompt)


# ---------- Main loop ----------
def main():
    print("Voice guide running. Ctrl+C to stop.")
    names, edges, start_spot = load_route()
    print(f"Route loaded: {sum(len(v) for v in edges.values())} directions, "
          f"{len(names)} destinations, trips start at '{start_spot}' if location is unknown")

    destination_id = read_json(DESTINATION_FILE, {}).get("id")   # don't re-announce an old destination
    here_id = read_json(HERE_FILE, {}).get("id")                 # ignore an old "I'm at"
    ask_id = read_json(ASK_FILE, {}).get("id")                   # ignore an old question

    destination = None
    known_spot = None             # best guess of where the user is (said, recognised, or counted)
    segment = None                # the walk we're following with the step counter
    visited = set()               # spots already passed on this trip
    last_nav_text = None
    last_nav_time = 0
    last_hazard_time = 0
    last_hazard_key = None
    last_stairs_time = 0
    safety_until = 0
    last_gemini_time = 0
    last_status_time = 0
    last_steps_seen = None

    def begin_trip(spot, opening):
        """Speak the first direction from `spot` and start following it with the step counter."""
        nonlocal segment, visited, last_nav_text, last_nav_time, known_spot, safety_until
        known_spot = spot
        visited = {spot}
        segment = None
        here_name = spoken_name(spot, names)
        dest_name = spoken_name(destination, names)
        if place_of(spot) == destination:
            last_nav_text = f"You are already at {dest_name}."
        else:
            path = find_path(spot, destination, edges)
            if path is None:
                last_nav_text = f"You are at {here_name}. I don't have a route from here to {dest_name}."
            else:
                segment = Segment(spot, path[1], edges[spot][path[1]],
                                  spoken_name(path[1], names), fresh_motion())
                last_nav_text = (f"{opening}Going to {dest_name}, about {path_steps(path, edges)} steps. "
                                 f"{edges[spot][path[1]]}")
        last_nav_time = time.time()
        speak(last_nav_text)
        safety_until = time.time() + SAFETY_GAP

    while True:
        now = time.time()
        p = read_json(PERCEPTION_FILE, {})
        waypoint = p.get("waypoint", "unknown")
        motion = fresh_motion()

        # Status line every 2 s: shows what VISTA thinks right now
        if now - last_status_time >= 2:
            last_status_time = now
            camera = f"camera: {waypoint} ({p.get('location_score', 0)})" if p else "camera: NO DATA"
            if motion:
                walked = segment.walked(motion) if segment else 0
                need = f"/{segment.steps_needed}" if segment else ""
                steps = f"steps {walked}{need}, turned {motion['yaw']:.0f}"
            else:
                steps = "step counter: OFF (no data from phone)"
            walk = f"{segment.frm} -> {segment.to}" if segment else "-"
            print(f"[STATUS] {camera} | {steps} | going to: {destination or 'nothing'} | walk: {walk}")

        # Remember where the user is from the camera when no trip is running
        if waypoint != "unknown" and segment is None:
            known_spot = waypoint

        # --- Destination chosen on the phone ("take me to the office") ---
        d = read_json(DESTINATION_FILE, {"to": None, "id": 0})
        if d.get("id") != destination_id:
            destination_id = d.get("id")
            destination = d.get("to") or None
            names, edges, start_spot = load_route()          # pick up route.json edits without restarting
            segment = None
            if destination:
                # "I'm at X" said in the same sentence arrives a moment later
                time.sleep(0.6)
                h = read_json(HERE_FILE, {})
                if h.get("id") != here_id and h.get("at"):
                    here_id = h.get("id")
                    begin_trip(h["at"], f"You are at {spoken_name(h['at'], names)}. ")
                elif known_spot:
                    begin_trip(known_spot, f"You are at {spoken_name(known_spot, names)}. ")
                else:
                    begin_trip(start_spot, f"Starting from {spoken_name(start_spot, names)}. ")
                if motion is None:
                    print("[WARNING] step counter is OFF - only the camera can guide. "
                          "Check the phone shows 'steps ...' at the bottom.")

        # --- User asked "Where am I?" ---
        a = read_json(ASK_FILE, {})
        if a.get("id") != ask_id:
            ask_id = a.get("id")
            if a.get("q") == "where":
                if segment and motion:
                    left = max(segment.steps_needed - segment.walked(motion), 0)
                    answer = (f"You are between {spoken_name(segment.frm, names)} and {segment.to_name}, "
                              f"about {left} steps from {segment.to_name}.")
                    if destination:
                        answer += f" You are going to {spoken_name(destination, names)}."
                elif known_spot:
                    answer = f"You are at {spoken_name(known_spot, names)}."
                    if not destination or place_of(known_spot) == destination:
                        answer += " Tap the screen and tell me where to go next."
                else:
                    answer = ("I'm not sure where you are. Tap the screen and say "
                              "I'm at, and the place name.")
                speak(answer)
                last_nav_time = now
                safety_until = now + SAFETY_GAP

        # --- User said where they are ("I'm at the entrance") ---
        h = read_json(HERE_FILE, {})
        if h.get("id") != here_id:
            here_id = h.get("id")
            if h.get("at"):
                if destination:
                    begin_trip(h["at"], f"You are at {spoken_name(h['at'], names)}. ")
                else:
                    known_spot = h["at"]
                    speak(f"You are at {spoken_name(h['at'], names)}. Tap the screen and tell me where to go.")

        # --- 1. Safety first: interrupts everything ---
        hazard_key = (p.get("hazard_object"), p.get("hazard_side"))
        if p.get("hazard_detected") and (hazard_key != last_hazard_key or now - last_hazard_time > HAZARD_REPEAT):
            obstacle = p.get("hazard_object") or "obstacle"
            side = p.get("hazard_side") or "ahead"
            speak(f"Careful, {obstacle} {side}.")             # "Careful, chair ahead."
            last_hazard_key = hazard_key
            last_hazard_time = now
            safety_until = now + SAFETY_GAP
        elif p.get("staircase_detected") and now - last_stairs_time > STAIRS_REPEAT:
            speak("Careful, stairs ahead.")
            last_stairs_time = now
            safety_until = now + SAFETY_GAP

        # --- 2. Step counter + turn sensor: the main guide during a trip ---
        elif segment and motion and now >= safety_until:
            sentence, reached, by_turning = segment.update(motion)
            if sentence:
                speak(sentence)
                last_nav_text = sentence
                last_nav_time = now
            elif reached:
                side = segment.side
                known_spot = reached
                visited.add(reached)
                here_name = spoken_name(reached, names)
                if place_of(reached) == destination:
                    extra = f" It is on your {side}." if side else ""
                    last_nav_text = f"You have arrived at {here_name}.{extra}"
                    segment = None
                else:
                    path = find_path(reached, destination, edges)
                    if path is None:
                        last_nav_text = f"You are at {here_name}. I don't have a route from here."
                        segment = None
                    else:
                        segment = Segment(reached, path[1], edges[reached][path[1]],
                                          spoken_name(path[1], names), motion)
                        start = "Good. " if by_turning else f"You are at {here_name}. "
                        last_nav_text = start + edges[reached][path[1]]
                last_nav_time = now
                speak(last_nav_text)
                if segment is None:
                    last_nav_text = None          # trip finished - don't keep repeating

        # --- 3. Camera: only guides when the step counter is OFF ---
        elif (destination and motion is None and now >= safety_until
              and waypoint != "unknown" and waypoint not in visited):
            visited.add(waypoint)
            begin_trip(waypoint, f"You are at {spoken_name(waypoint, names)}. ")

        # --- 4. Repeat the direction if the user is standing still ---
        elif destination and last_nav_text and now >= safety_until and now - last_nav_time > NAV_REPEAT:
            steps_now = motion["steps"] if motion else None
            if steps_now is None or steps_now == last_steps_seen:
                speak(last_nav_text)
            last_nav_time = now
            last_steps_seen = steps_now

        # --- 5. No destination: occasional Gemini scene description ---
        elif client and not destination and now >= safety_until and now - last_gemini_time > GEMINI_EVERY:
            last_gemini_time = now
            try:
                speak(gemini_scene_sentence())
            except Exception as e:
                print(f"Gemini error: {e}")

        time.sleep(0.3)


if __name__ == "__main__":
    main()