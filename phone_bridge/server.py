import os
import json
import time
from flask import Flask, request, jsonify, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
FRAME_FILE = os.path.join(ROOT, "perception", "latest_frame.jpg")
SPEECH_FILE = os.path.join(ROOT, "assistant", "latest_speech.json")

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, ssl_context="adhoc")