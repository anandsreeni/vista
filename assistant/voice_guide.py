import json
import os
import time
import logging
import pyttsx3
from google import genai

logging.getLogger("google_genai").setLevel(logging.ERROR)

SCENE_GRAPH_FILE = os.path.join("..", "scene_graph", "scene_graph.json")

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-flash-lite-latest"
FALLBACK_MODEL = "gemini-2.5-flash-lite"

MIN_SECONDS_BETWEEN_SPEECH = 15

SYSTEM_INSTRUCTION = """You are VISTA, a calm walking guide for a visually impaired user.
You will be given the user's current location and the most recent landmark the camera noticed.
Respond with ONE short spoken sentence guiding or informing them — like a real-time narrator, not a report.
If a HAZARD or STAIRCASE appears in the landmark, prioritize warning about it clearly and calmly.
Never invent details that weren't given to you."""

def load_scene_graph():
    if not os.path.exists(SCENE_GRAPH_FILE):
        return {"nodes": [], "current_location": "unknown"}
    try:
        with open(SCENE_GRAPH_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"nodes": [], "current_location": "unknown"}

def get_latest_landmark(graph):
    nodes = graph.get("nodes", [])
    if not nodes:
        return None
    return max(nodes, key=lambda n: n.get("last_seen", ""))

def build_prompt(location, landmark):
    if landmark:
        landmark_text = f"{landmark['name']} (seen at '{landmark['location']}')"
    else:
        landmark_text = "nothing notable yet"
    return f"Current location: {location}\nMost recent landmark: {landmark_text}"

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
                if "429" in msg:
                    print(f"{model_name} quota exhausted, trying fallback...")
                    break
                if "404" in msg:
                    print(f"{model_name} not available, trying fallback...")
                    break
                if "503" in msg and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"{model_name} overloaded, retrying in {wait}s...")
                    time.sleep(wait)
                elif "503" in msg:
                    print(f"{model_name} still overloaded, trying fallback...")
                    break
                else:
                    raise
    raise RuntimeError("All Gemini models unavailable right now.")

def speak(text):
    print(f"VISTA (spoken): {text}")
    local_engine = pyttsx3.init()
    local_engine.setProperty("rate", 170)
    local_engine.say(text)
    local_engine.runAndWait()
    local_engine.stop()

def main():
    print("Voice guide running. Ctrl+C to stop.")
    last_signature = None
    last_spoken_time = 0

    while True:
        graph = load_scene_graph()
        location = graph.get("current_location", "unknown")
        landmark = get_latest_landmark(graph)

        signature = (location, landmark["name"] if landmark else None)

        now = time.time()
        enough_time_passed = (now - last_spoken_time) >= MIN_SECONDS_BETWEEN_SPEECH

        if signature != last_signature and enough_time_passed:
            last_signature = signature
            prompt_text = build_prompt(location, landmark)
            try:
                instruction = ask_gemini(prompt_text)
                speak(instruction)
                last_spoken_time = now
            except Exception as e:
                print(f"Gemini error: {e}")

        time.sleep(2)

if __name__ == "__main__":
    main()