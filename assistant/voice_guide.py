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

# ---------- Timing ----------
HAZARD_REPEAT = 6        # seconds before the same safety warning is repeated
STAIRS_REPEAT = 8
NAV_REPEAT = 30          # repeat the current direction if the user hasn't reached a new spot
SAFETY_GAP = 2.5         # let a warning finish before a direction is spoken
GEMINI_EVERY = 20        # scene description only when NO destination is set

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
    return route.get("names", {}), edges


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


def direction_message(waypoint, destination, names, edges):
    here = spoken_name(waypoint, names)
    if not destination:
        return f"You are at {here}."
    if place_of(waypoint) == destination:
        return f"You have arrived at {here}."
    path = find_path(waypoint, destination, edges)
    if path is None:
        return f"You are at {here}. I don't have a route from here to {spoken_name(destination, names)}."
    next_spot = path[1]
    return f"You are at {here}. {edges[waypoint][next_spot]}"


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
    names, edges = load_route()
    print(f"Route loaded: {sum(len(v) for v in edges.values())} directions, {len(names)} destinations")

    last_waypoint = None
    last_nav_text = None
    last_nav_time = 0
    last_hazard_time = 0
    last_stairs_time = 0
    safety_until = 0
    last_gemini_time = 0
    destination = None
    destination_id = None
    last_status_time = 0
    here_id = read_json(HERE_FILE, {}).get("id")   # ignore an old message from before we started

    while True:
        now = time.time()
        p = read_json(PERCEPTION_FILE, {})
        waypoint = p.get("waypoint", "unknown")

        # Status line every 2 s: shows what VISTA thinks right now
        if now - last_status_time >= 2:
            last_status_time = now
            if not p:
                print("[STATUS] no data from perception.py - is it running and getting frames?")
            else:
                print(f"[STATUS] I see: {waypoint} (score {p.get('location_score', 0)})"
                      f"  |  going to: {destination or 'nothing'}")

        # --- Destination chosen on the phone ---
        d = read_json(DESTINATION_FILE, {"to": None, "id": 0})
        if d.get("id") != destination_id:
            destination_id = d.get("id")
            destination = d.get("to") or None
            names, edges = load_route()          # pick up route.json edits without restarting
            if destination and waypoint == "unknown":
                speak(f"Going to {spoken_name(destination, names)}. Start walking. "
                      "I will guide you as soon as I recognise where you are.")
                safety_until = now + 5
            elif destination:
                speak(f"Going to {spoken_name(destination, names)}.")
                safety_until = now + SAFETY_GAP
            last_waypoint = None                 # give a fresh direction from where we are

        # --- User said where they are ("I'm at the entrance") ---
        h = read_json(HERE_FILE, {})
        if h.get("id") != here_id:
            here_id = h.get("id")
            if h.get("at"):
                last_waypoint = h["at"]
                last_nav_text = direction_message(h["at"], destination, names, edges)
                last_nav_time = now
                speak(last_nav_text)
                safety_until = now + SAFETY_GAP

        # --- 1. Safety first: interrupts everything ---
        if p.get("hazard_detected") and now - last_hazard_time > HAZARD_REPEAT:
            speak("Careful, obstacle ahead.")
            last_hazard_time = now
            safety_until = now + SAFETY_GAP
        elif p.get("staircase_detected") and now - last_stairs_time > STAIRS_REPEAT:
            speak("Careful, stairs ahead.")
            last_stairs_time = now
            safety_until = now + SAFETY_GAP

        # --- 2. Directions: the moment a new spot is recognised ---
        elif now >= safety_until and waypoint != "unknown" and waypoint != last_waypoint:
            last_waypoint = waypoint
            last_nav_text = direction_message(waypoint, destination, names, edges)
            last_nav_time = now
            speak(last_nav_text)

        # --- 3. Repeat the direction if the user is taking a while ---
        elif destination and last_nav_text and now >= safety_until and now - last_nav_time > NAV_REPEAT:
            last_nav_time = now
            speak(last_nav_text)

        # --- 4. No destination: occasional Gemini scene description ---
        elif client and not destination and now >= safety_until and now - last_gemini_time > GEMINI_EVERY:
            last_gemini_time = now
            try:
                speak(gemini_scene_sentence())
            except Exception as e:
                print(f"Gemini error: {e}")

        time.sleep(0.3)


if __name__ == "__main__":
    main()