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

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")


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

    # 2. Reply with the latest sentence for the phone to speak
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, ssl_context="adhoc")