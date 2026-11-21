"""Shared 2D world for the Spot nav demo (env origin = world origin). Box: (name, cx, cy, sx, sy, height, spawn_here).
Zig-zag course (2026-11-08): a divider splits the room; 3 baffles make the start->bench leg zig-zag (north half),
2 baffles make the bench->bin leg zig-zag (south half). Bench and pick geometry unchanged."""
SHIFT = 1.5                                                     # bench moved 1.5 m forward (as in full_demo)
BENCH = ("bench", 0.85 + SHIFT, 0.0, 0.6, 1.2, 0.40, False)     # spawned by the task itself
BIN = ("bin", -3.5, -2.8, 0.317, 0.475, 0.146, False)           # spawned by the demo (KLT, scale 1.6)
# ---- optional scene override: nav/scene.json (or $NAV_SCENE) - read by Isaac AND gen_map, so sim and map always match ----
import json as _json, os as _os
SCENE_FILE = _os.environ.get("NAV_SCENE", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "scene.json"))
SCENE = _json.load(open(SCENE_FILE)) if _os.path.exists(SCENE_FILE) else {}
if "bench" in SCENE:                       # [x, y, "y"|"x"] = long side (1.2 m) along y (default) or along x
    _bx, _by, _bo = SCENE["bench"]
    _sx, _sy = (0.6, 1.2) if _bo == "y" else (1.2, 0.6)
    BENCH = ("bench", float(_bx), float(_by), _sx, _sy, 0.40, False)
if "bin" in SCENE:                         # [x, y]
    BIN = ("bin", float(SCENE["bin"][0]), float(SCENE["bin"][1]), 0.317, 0.475, 0.146, False)
ROOM_X, ROOM_Y, WALL_T, WALL_H = (-4.8, 4.5), (-3.8, 3.8), 0.10, 1.0
WALLS = [
    ("wall_s", sum(ROOM_X) / 2, ROOM_Y[0], ROOM_X[1] - ROOM_X[0], WALL_T, WALL_H, True),
    ("wall_n", sum(ROOM_X) / 2, ROOM_Y[1], ROOM_X[1] - ROOM_X[0], WALL_T, WALL_H, True),
    ("wall_w", ROOM_X[0], sum(ROOM_Y) / 2, WALL_T, ROOM_Y[1] - ROOM_Y[0], WALL_H, True),
    ("wall_e", ROOM_X[1], sum(ROOM_Y) / 2, WALL_T, ROOM_Y[1] - ROOM_Y[0], WALL_H, True),
]
BT, BH = 0.15, 0.8                                              # baffle thickness / height


def vwall(name, x, y0, y1):
    return (name, x, (y0 + y1) / 2, BT, abs(y1 - y0), BH, True)


def hwall(name, y, x0, x1):
    return (name, (x0 + x1) / 2, y, abs(x1 - x0), BT, BH, True)


DIVIDER = hwall("divider", 0.0, ROOM_X[0], 0.0)                 # north half = leg 1, south half = leg 2
NB1 = vwall("baffle_n1", -2.95, 0.0, 2.0)                       # leg 1: over the top ...
NB2 = vwall("baffle_n2", -1.45, 1.5, ROOM_Y[1])                 # ... under ...
NB3 = vwall("baffle_n3", 0.0, 0.0, 2.0)                         # ... over the top, then down to the bench
SB1 = vwall("baffle_s1", -0.7, -2.0, 0.0)                       # leg 2: under ...
SB2 = vwall("baffle_s2", -2.1, ROOM_Y[0], -1.5)                 # ... over the top, then down to the bin
EXTRA_BINS = [(str(_n), float(_v[0]), float(_v[1]), 0.317, 0.475, 0.146, False)   # PLACE: more bins (scene.json "bins")
              for _n, _v in SCENE.get("bins", {}).items()]
OBSTACLES = [BENCH, BIN, DIVIDER, NB1, NB2, NB3, SB1, SB2] + WALLS + EXTRA_BINS

START = (-3.88, 1.0, 0.0)                                       # robot start x, y, yaw_deg (west lane of leg 1)

# Pick approach (arm_models_for_nav2.md): cube at bench centre, best stand-off 0.85-0.90 m, Nav2 stops 0.5 m earlier.
CUBE_XY = (BENCH[1], 0.0)
STANDOFF = 0.875
PREPICK_EXTRA = 0.5
PICK_POSE = (CUBE_XY[0] - STANDOFF, CUBE_XY[1], 0.0)
PREPICK_POSE = (CUBE_XY[0] - STANDOFF - PREPICK_EXTRA, CUBE_XY[1], 0.0)

# Surprise obstacle: spawned in the sim but NOT in OBSTACLES, so it is NOT in the Nav2 map.
# Nav2 can only avoid it through the scan (PhysX lidar). Placed on the leg-1 route east of baffle_n3.
SURPRISE = [("surprise_a", 0.85, 1.3, 0.4, 0.4, 0.6, True)]

# scene overrides that must come after the defaults above
if "start" in SCENE:                       # [x, y] (robot always starts facing +x)
    START = (float(SCENE["start"][0]), float(SCENE["start"][1]), 0.0)
if "surprise" in SCENE:                    # list of [name, x, y, sx, sy, h]; [] = none
    SURPRISE = [(str(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5]), True) for b in SCENE["surprise"]]

# Moving obstacle (NOT in the Nav2 map, only the lidar sees it): during `phase`, when Spot comes within `trigger` m of
# `target`, the box slides from `start` to `target` at `speed` m/s; it pauses while closer than `keep_away` m to Spot.
MOVER = {"size": (0.5, 0.5, 0.6), "start": (2.6, -1.6), "target": (0.75, -1.6), "phase": "NAV_BIN",
         "trigger": 2.2, "speed": 0.4, "keep_away": 0.9}
if "mover" in SCENE:                       # null = no moving obstacle, or a dict like MOVER
    MOVER = None if SCENE["mover"] is None else {**MOVER, **SCENE["mover"]}
