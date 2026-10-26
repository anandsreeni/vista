import os
import re
import json
import time
from flask import Flask, request, jsonify, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
FRAME_FILE = os.path.join(ROOT, "perception", "latest_frame.jpg")
SPEECH_FILE = os.path.join(ROOT, "assistant", "latest_speech.json")
WAYPOINT_DIR = os.path.join(ROOT, "perception", "waypoints")
MOTION_FILE = os.path.join(ROOT, "assistant", "motion.json")

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.route("/sensor")
def sensor_test():
    return send_from_directory(BASE, "sensor.html")


@app.route("/frame", methods=["POST"])
def frame():
    # 1. Save the phone's frame (write to .tmp, then swap, so perception never reads half a file)
    tmp = FRAME_FILE + ".tmp"
    with open(tmp, "wb") as f:
        f.write(request.get_data())
    for attempt in range(5):
        try:
            os.replace(tmp, FRAME_FILE)
            break
        except PermissionError:      # Windows: perception.py has the file open
            time.sleep(0.02)

    # 2. Step count + turn from the phone's motion sensor (voice_guide.py reads this)
    if "X-Steps" in request.headers:
        try:
            motion = {
                "steps": int(request.headers["X-Steps"]),
                "yaw": float(request.headers.get("X-Yaw", 0)),
                "t": time.time(),
            }
            tmp = MOTION_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(motion, f)
            for attempt in range(5):
                try:
                    os.replace(tmp, MOTION_FILE)
                    break
                except PermissionError:
                    time.sleep(0.02)
        except ValueError:
            pass

    # 3. Reply with the latest sentence for the phone to speak
    speech = {"id": 0, "text": ""}
    try:
        with open(SPEECH_FILE, "r") as f:
            speech = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return jsonify(speech)


@app.route("/capture", methods=["POST"])
def capture():
    # Mapping mode: save one photo into perception/waypoints/<name>/
    # "Room 204" -> "room_204"; anything that isn't a letter/number/underscore is dropped
    name = request.args.get("name", "").strip().lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    if not name:
        return jsonify({"ok": False, "error": "Type a waypoint name first"}), 400

    folder = os.path.join(WAYPOINT_DIR, name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{int(time.time() * 1000)}.jpg")
    with open(path, "wb") as f:
        f.write(request.get_data())

    count = len([x for x in os.listdir(folder) if x.endswith(".jpg")])
    print(f"[MAP] saved {path}  ({count} photos for '{name}')")
    return jsonify({"ok": True, "name": name, "count": count})


@app.route("/undo", methods=["POST"])
def undo():
    # Mapping mode: delete the most recent photo of a waypoint (blurry shot, someone walked past...)
    name = request.args.get("name", "").strip().lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    folder = os.path.join(WAYPOINT_DIR, name)
    if not name or not os.path.isdir(folder):
        return jsonify({"ok": False, "error": "Nothing to undo"}), 400

    photos = sorted(x for x in os.listdir(folder) if x.endswith(".jpg"))
    if not photos:
        return jsonify({"ok": False, "error": "Nothing to undo"}), 400
    os.remove(os.path.join(folder, photos[-1]))
    print(f"[MAP] deleted {photos[-1]} from '{name}'")
    return jsonify({"ok": True, "name": name, "count": len(photos) - 1})


ROUTE_FILE = os.path.join(ROOT, "assistant", "route.json")
DESTINATION_FILE = os.path.join(ROOT, "assistant", "destination.json")


@app.route("/destinations")
def destinations():
    # Places for the phone's "Where to?" list = the "names" in assistant/route.json
    try:
        with open(ROUTE_FILE, "r") as f:
            names = json.load(f).get("names", {})
    except (FileNotFoundError, json.JSONDecodeError):
        names = {}
    places = [{"id": k, "name": v} for k, v in sorted(names.items())]
    return jsonify({"destinations": places})


@app.route("/destination", methods=["POST"])
def set_destination():
    # Phone picked a destination ("" = just tell me where I am). voice_guide.py reads this file.
    to = request.args.get("to", "")
    with open(DESTINATION_FILE, "w") as f:
        json.dump({"to": to, "id": int(time.time() * 1000)}, f)
    print(f"[NAV] destination set to: {to or '(none)'}")
    return jsonify({"ok": True, "to": to})


HERE_FILE = os.path.join(ROOT, "assistant", "here.json")


ASK_FILE = os.path.join(ROOT, "assistant", "ask.json")


@app.route("/ask", methods=["POST"])
def ask():
    # User asked a question by voice, e.g. "where am I?". voice_guide.py answers it.
    q = request.args.get("q", "")
    with open(ASK_FILE, "w") as f:
        json.dump({"q": q, "id": int(time.time() * 1000)}, f)
    print(f"[ASK] {q}")
    return jsonify({"ok": True})


@app.route("/here", methods=["POST"])
def set_here():
    # User said "I'm at the entrance". voice_guide.py reads this file.
    at = request.args.get("at", "")
    with open(HERE_FILE, "w") as f:
        json.dump({"at": at, "id": int(time.time() * 1000)}, f)
    print(f"[NAV] user says they are at: {at}")
    return jsonify({"ok": True, "at": at})


LEGS_FILE = os.path.join(ROOT, "assistant", "mapped_legs.json")


@app.route("/leg", methods=["POST"])
def save_leg():
    # Map mode: the walk between two spots, measured by the phone (steps + turn)
    leg = {
        "from": request.args.get("from", ""),
        "to": request.args.get("to", ""),
        "steps": int(request.args.get("steps", 0)),
        "yaw": float(request.args.get("yaw", 0)),
        "t": time.time(),
    }
    try:
        with open(LEGS_FILE, "r") as f:
            legs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        legs = []
    legs.append(leg)
    with open(LEGS_FILE, "w") as f:
        json.dump(legs, f, indent=1)
    print(f"[MAP] walk {leg['from']} -> {leg['to']}: {leg['steps']} steps, turned {leg['yaw']:.0f}")
    return jsonify({"ok": True, "count": len(legs)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, ssl_context="adhoc")