import json
import os
import time
from collections import Counter
from datetime import datetime
from datetime import datetime, timedelta

GRAPH_FILE = "scene_graph.json"
PERCEPTION_FILE = "../perception/latest_perception.json"

seen_counts = Counter()

def load_graph():
    if os.path.exists(GRAPH_FILE):
        with open(GRAPH_FILE, "r") as f:
            return json.load(f)
    return {"nodes": []}

def save_graph(graph):
    with open(GRAPH_FILE, "w") as f:
        json.dump(graph, f, indent=2)

def add_object(graph, name, location="unknown", threshold=5):
    seen_counts[name] += 1
    if seen_counts[name] < threshold:
        return graph  # not seen enough times yet, ignore for now

    for node in graph["nodes"]:
        if node["name"] == name and node["location"] == location:
            node["last_seen"] = str(datetime.now())
            node["times_seen"] = node.get("times_seen", 1) + 1
            return graph

    graph["nodes"].append({
        "name": name,
        "location": location,
        "first_seen": str(datetime.now()),
        "last_seen": str(datetime.now()),
        "times_seen": 1
    })
    return graph

def clean_stale_nodes(graph, min_sightings=10, max_age_minutes=10):
    now = datetime.now()
    kept_nodes = []
    for node in graph["nodes"]:
        last_seen = datetime.fromisoformat(node["last_seen"])
        age = now - last_seen
        if node["times_seen"] < min_sightings and age > timedelta(minutes=max_age_minutes):
            continue  # skip = remove this node
        kept_nodes.append(node)
    graph["nodes"] = kept_nodes
    return graph

def watch_perception():
    graph = load_graph()
    print("Watching for Person 1's output... (Ctrl+C to stop)")
    loop_count = 0

    while True:
        try:
            with open(PERCEPTION_FILE, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.5)
            continue

        for obj_name in data.get("objects", []):
            graph = add_object(graph, obj_name, location="current_spot")

        if data.get("hazard_detected"):
            graph = add_object(graph, "HAZARD", location="current_spot", threshold=1)

        loop_count += 1
        if loop_count % 100 == 0:
            graph = clean_stale_nodes(graph)
            print(">>> Ran cleanup, removed stale low-confidence nodes")

        save_graph(graph)
        print(graph)
        time.sleep(0.5)

if __name__ == "__main__":
    watch_perception()