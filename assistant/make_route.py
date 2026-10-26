"""
Builds route.json from the walks measured in map mode (mapped_legs.json).

Run from the assistant folder, after mapping:
    python make_route.py

- Keeps your old route.json as a backup (route_backup_<time>.json)
- Keeps any directions you wrote by hand that weren't re-measured
- A turn measured at the START of a walk becomes "then turn left/right" at the END of the walk before it
"""
import os
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LEGS_FILE = os.path.join(HERE, "mapped_legs.json")
ROUTE_FILE = os.path.join(HERE, "route.json")
TURN_DEG = 50            # turned more than this = a real turn


def turn_word(yaw):
    if yaw > TURN_DEG:
        return "left"
    if yaw < -TURN_DEG:
        return "right"
    return None


def auto_name(spot):
    """'f0_computer_lab_1' -> 'computer lab 1', 'f0_office' -> 'the office'. None for junctions/stairs."""
    place = spot.split("__")[0]
    if "_" in place and place[0] == "f" and place[1].isdigit():
        place = place.split("_", 1)[1]
    if any(w in place for w in ("junction", "stairs", "lift", "corridor", "turn")):
        return None
    text = place.replace("_", " ")
    if text.startswith(("room ", "lab ", "computer lab")):
        return text
    return "the " + text


def main():
    try:
        with open(LEGS_FILE, "r") as f:
            legs = json.load(f)
    except FileNotFoundError:
        print("No mapped_legs.json yet - map some walks in map mode first.")
        return
    legs.sort(key=lambda l: l["t"])

    # latest measurement wins if a walk was mapped twice
    latest = {}
    for i, leg in enumerate(legs):
        latest[(leg["from"], leg["to"])] = i
    keep = sorted(latest.values())
    legs = [legs[i] for i in keep]

    # A walk's turn happens at its START (you turn, then walk). Move it to the END of the walk
    # that arrives at that spot, so VISTA can say "In 3 steps, turn left" before you get there.
    def is_back(spot):
        return "__" in spot

    end_turn = {}             # index of arriving walk -> "left"/"right"
    turn_used = set()         # walks whose turn was moved
    for i, leg in enumerate(legs):
        word = turn_word(leg["yaw"])
        if not word:
            continue
        for j, before in enumerate(legs):
            # arrives where this walk starts, same direction of travel, and isn't just coming back from there
            if (j != i and before["to"] == leg["from"] and is_back(before["to"]) == is_back(leg["to"])
                    and before["from"].split("__")[0] != leg["to"].split("__")[0]):
                end_turn[j] = word
                turn_used.add(i)

    new_edges = []
    for i, leg in enumerate(legs):
        n = leg["steps"]
        word = turn_word(leg["yaw"])
        if word and i not in turn_used:
            say = f"Turn {word}, then walk straight about {n} steps."     # turn right at the start
        else:
            say = f"Walk straight about {n} steps"
            say += f", then turn {end_turn[i]}." if i in end_turn else "."
        new_edges.append({"from": leg["from"], "to": leg["to"], "say": say})

    # merge with the existing route.json
    try:
        with open(ROUTE_FILE, "r") as f:
            route = json.load(f)
        backup = os.path.join(HERE, f"route_backup_{time.strftime('%H%M%S')}.json")
        with open(backup, "w") as f:
            json.dump(route, f, indent=2)
        print(f"Old route.json saved as {os.path.basename(backup)}")
    except (FileNotFoundError, json.JSONDecodeError):
        route = {}

    measured = {(e["from"], e["to"]) for e in new_edges}
    kept = [e for e in route.get("edges", []) if (e["from"], e["to"]) not in measured]
    names = route.get("names", {})
    for e in new_edges:
        for spot in (e["from"], e["to"]):
            base = spot.split("__")[0]
            if base not in names and auto_name(base):
                names[base] = auto_name(base)

    route["start"] = route.get("start") or legs[0]["from"]
    route["names"] = names
    route["edges"] = new_edges + kept

    with open(ROUTE_FILE, "w") as f:
        f.write("{\n")
        f.write(f'  "start": {json.dumps(route["start"])},\n\n')
        f.write('  "names": ' + json.dumps(names, indent=4).replace("\n}", "\n  }") + ",\n\n")
        f.write('  "edges": [\n')
        lines = [f'    {{"from": {json.dumps(e["from"])}, "to": {json.dumps(e["to"])}, "say": {json.dumps(e["say"])}}}'
                 for e in route["edges"]]
        f.write(",\n".join(lines))
        f.write("\n  ]\n}\n")

    print(f"\nroute.json written: {len(new_edges)} measured walks, {len(kept)} kept from before\n")
    for e in new_edges:
        print(f"  {e['from']:22} -> {e['to']:22} {e['say']}")
    print("\nPlaces in the 'where to' list:")
    for k, v in names.items():
        print(f"  {k:22} = {v}")
    print("\nRestart voice_guide.py to use the new route.")


if __name__ == "__main__":
    main()