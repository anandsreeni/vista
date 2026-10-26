"""
Fixes the waypoint folder names and checks they match assistant/route.json.

Run from the perception folder:
    python fix_names.py
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
WAYPOINT_DIR = os.path.join(HERE, "waypoints")
ROUTE_FILE = os.path.join(os.path.dirname(HERE), "assistant", "route.json")

# Long names from mapping day (directions were typed into the name box) -> correct spot name.
# The photos in each folder are of the place you walked TO (the second name).
RENAME = {
    "f0_entrance_f0_junction_1_straight_13_steps_turn_left": "f0_junction_1",
    "f0_junction_1__f0_office_straight_17_steps__door_on_left": "f0_office",
    "f0_officef0_junction_2straight_25_steps_turn_left": "f0_junction_2",
    "f0_officef0_junction_2straight_25_steps_turn_left_f0_women_washroom": "f0_washroom_women",
    "f0_women_washroom___f0_room_104straight_35_steps_door_on_left": "f0_room_104",
    "f0_room_104__f0_staffroomstraight_30_steps_door_on_left": "f0_staffroom",
    "f0_staffroomf0_entrance_insidestraight_20_steps__entrance_on_left": "f0_entrance__inside",
    "f0_entrance_insidef0_outsidestraight_15_steps__out_through_the_main_door": "f0_outside",
    # renamed by hand earlier with one underscore
    "f0_entrance_inside": "f0_entrance__inside",
    "f0_women_washroom": "f0_washroom_women",
}


def photos_in(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]


def main():
    if not os.path.isdir(WAYPOINT_DIR):
        print(f"[ERROR] No waypoints folder at {WAYPOINT_DIR}")
        return

    # ---------- 1. Rename ----------
    print("\n===== 1. FIXING FOLDER NAMES =====")
    changed = 0
    for old, new in RENAME.items():
        old_path = os.path.join(WAYPOINT_DIR, old)
        new_path = os.path.join(WAYPOINT_DIR, new)
        if not os.path.isdir(old_path):
            continue
        if os.path.isdir(new_path):
            # target already exists: move the photos in (merge)
            for f in photos_in(old_path):
                os.replace(os.path.join(old_path, f), os.path.join(new_path, f))
            os.rmdir(old_path)
            print(f"[MERGED]  {old}\n          -> {new}")
        else:
            os.rename(old_path, new_path)
            print(f"[RENAMED] {old}\n          -> {new}")
        changed += 1
    if changed == 0:
        print("Nothing to rename.")

    # ---------- 2. Check against route.json ----------
    folders = {}
    for name in sorted(os.listdir(WAYPOINT_DIR)):
        path = os.path.join(WAYPOINT_DIR, name)
        if os.path.isdir(path):
            folders[name] = len(photos_in(path))

    try:
        with open(ROUTE_FILE, "r") as f:
            route = json.load(f)
    except FileNotFoundError:
        print(f"\n[ERROR] route.json not found at {ROUTE_FILE}")
        return
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] route.json has a typing mistake: {e}")
        return

    in_route = set()
    for e in route.get("edges", []):
        in_route.add(e["from"])
        in_route.add(e["to"])

    print("\n===== 2. YOUR SPOTS =====")
    for name, count in folders.items():
        if count < 4:
            note = "  <-- only a few photos, RECORD again"
        elif name not in in_route:
            note = "  <-- not used in route.json (VISTA will say 'no route' here)"
        else:
            note = "  OK"
        print(f"{name:28} {count:3} photos{note}")

    missing = sorted(in_route - set(folders))
    print("\n===== 3. IN route.json BUT NO PHOTOS (VISTA can never recognise these) =====")
    if missing:
        for name in missing:
            print(f"  {name}   -> Map mode, type '{name}', RECORD")
    else:
        print("  None - every spot in route.json has photos.")
    print("\nNow restart perception.py so it loads the new names.\n")


if __name__ == "__main__":
    main()