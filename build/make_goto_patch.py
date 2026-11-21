"""General go_to: stand/pre-pose planner with map checks, movable scenes (scene.json), demo uses planned poses.
Builds full_demo_goto.py from full_demo_perc_nav_amcl.py. Safe to run twice."""
import json, os, sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
N = f"{C}/nav"

# 1) nav_world: optional scene.json overrides (bench position + orientation, bin, start, surprise)
p = f"{N}/nav_world.py"
s = open(p).read()
if "SCENE_FILE" not in s:
    o = 'BIN = ("bin", -3.5, -2.8, 0.317, 0.475, 0.146, False)           # spawned by the demo (KLT, scale 1.6)\n'
    assert s.count(o) == 1, "nav_world BIN line not found"
    s = s.replace(o, o +
        '# ---- optional scene override: nav/scene.json (or $NAV_SCENE) - read by Isaac AND gen_map, so sim and map always match ----\n'
        'import json as _json, os as _os\n'
        'SCENE_FILE = _os.environ.get("NAV_SCENE", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "scene.json"))\n'
        'SCENE = _json.load(open(SCENE_FILE)) if _os.path.exists(SCENE_FILE) else {}\n'
        'if "bench" in SCENE:                       # [x, y, "y"|"x"] = long side (1.2 m) along y (default) or along x\n'
        '    _bx, _by, _bo = SCENE["bench"]\n'
        '    _sx, _sy = (0.6, 1.2) if _bo == "y" else (1.2, 0.6)\n'
        '    BENCH = ("bench", float(_bx), float(_by), _sx, _sy, 0.40, False)\n'
        'if "bin" in SCENE:                         # [x, y]\n'
        '    BIN = ("bin", float(SCENE["bin"][0]), float(SCENE["bin"][1]), 0.317, 0.475, 0.146, False)\n')
    s += ('\n# scene overrides that must come after the defaults above\n'
          'if "start" in SCENE:                       # [x, y] (robot always starts facing +x)\n'
          '    START = (float(SCENE["start"][0]), float(SCENE["start"][1]), 0.0)\n'
          'if "surprise" in SCENE:                    # list of [name, x, y, sx, sy, h]; [] = none\n'
          '    SURPRISE = [(str(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5]), True) for b in SCENE["surprise"]]\n')
    open(p, "w").write(s)
    print("[PATCH] nav_world.py: scene.json overrides added")
else:
    print("[PATCH] nav_world.py already has scene overrides")

# 2) goto_planner.py (pure python + numpy; usable from Isaac, system python, and the perception side)
open(f"{N}/goto_planner.py", "w").write('''"""General go_to planning for Spot: stand pose + pre-pose for any target and approach direction, checked against the map.
Pure python + numpy. Usable from Isaac (py3.12), system python, and the perception/manipulation side.
  stand pose : where Spot must finally stand (target `standoff` m straight ahead), reached by the scripted fine approach
  pre-pose   : Nav2 goal, `pre_dist` m behind the stand pose on the same heading; must leave room to turn in place
plan() tries each candidate heading, rejects unsafe ones with a reason, and returns the valid one closest to the robot."""
import math
import numpy as np

HALF_L, HALF_W = 0.55, 0.25          # Spot footprint half-length / half-width (arm stowed), same as Nav2


class GotoPlanner:
    def __init__(self, obstacles, pre_dist=0.5, turn_r=0.65, max_extra=0.5):
        """obstacles: nav_world-style boxes (name, cx, cy, sx, sy, h, spawn). turn_r: centre-to-obstacle distance needed
        to turn in place (corner radius 0.60 + margin). max_extra: how much further back the pre-pose may move to find room."""
        self.boxes = [(n, cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2) for n, cx, cy, sx, sy, *_ in obstacles]
        self.pre_dist, self.turn_r, self.max_extra = pre_dist, turn_r, max_extra

    def dist(self, x, y):
        """Distance from a point to the nearest obstacle (0 inside one) and its name."""
        best, who = math.inf, None
        for n, x0, x1, y0, y1 in self.boxes:
            d = math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))
            if d < best:
                best, who = d, n
        return best, who

    def fp_clear(self, x, y, yaw):
        """Clearance of Spot's footprint outline at (x, y, yaw); negative = overlapping an obstacle."""
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [(HALF_L, HALF_W), (HALF_L, -HALF_W), (-HALF_L, -HALF_W), (-HALF_L, HALF_W)]
        best, who = math.inf, None
        for k in range(4):
            (ax, ay), (bx, by) = corners[k], corners[(k + 1) % 4]
            for t in np.linspace(0.0, 1.0, 12):
                px, py = ax + (bx - ax) * t, ay + (by - ay) * t
                d, n = self.dist(x + c * px - s * py, y + s * px + c * py)
                if d < best:
                    best, who = d, n
        if best == 0.0:
            best = -0.01
        return best, who

    def plan(self, target_xy, standoff, headings, robot_xy=None, stand_clear=-0.03, label=""):
        """Returns dict(ok, stand=(x, y, yaw), pre=(x, y, yaw), reason, tried=[...]).
        headings: candidate approach directions (rad) = direction Spot faces at the stand pose."""
        tx, ty = target_xy
        tried, valid = [], []
        for h in headings:
            ux, uy = math.cos(h), math.sin(h)
            sx, sy = tx - standoff * ux, ty - standoff * uy
            c_stand, who = self.fp_clear(sx, sy, h)
            tag = f"heading {math.degrees(h):+4.0f}"
            if c_stand < stand_clear:
                tried.append(f"{tag}: stand pose overlaps {who} by {-c_stand * 100:.0f} cm"); continue
            ok_pre = None
            for extra in np.arange(0.0, self.max_extra + 1e-6, 0.1):
                px, py = sx - (self.pre_dist + extra) * ux, sy - (self.pre_dist + extra) * uy
                d_pre, who_pre = self.dist(px, py)
                corridor = min(self.fp_clear(sx + (px - sx) * t, sy + (py - sy) * t, h)[0] for t in np.linspace(0, 1, 8))
                if d_pre >= self.turn_r and corridor >= stand_clear:
                    ok_pre = (px, py, extra); break
            if ok_pre is None:
                tried.append(f"{tag}: no room to turn behind the stand pose (nearest {who_pre} {d_pre:.2f} m < {self.turn_r})"); continue
            px, py, extra = ok_pre
            far = math.hypot(px - robot_xy[0], py - robot_xy[1]) if robot_xy is not None else 0.0
            tried.append(f"{tag}: OK (stand clearance {c_stand * 100:+.0f} cm, pre-pose {self.pre_dist + extra:.1f} m back, {far:.1f} m from robot)")
            valid.append((far, (sx, sy, h), (px, py, h)))
        if not valid:
            return {"ok": False, "stand": None, "pre": None, "tried": tried,
                    "reason": f"{label}: no reachable approach to ({tx:+.2f},{ty:+.2f})"}
        far, stand, pre = min(valid, key=lambda v: v[0])
        return {"ok": True, "stand": stand, "pre": pre, "tried": tried,
                "reason": f"{label}: approach heading {math.degrees(stand[2]):+.0f} deg"}


def pick_headings(bench):
    """Approach from either long side of the bench (the short sides leave Spot's nose over the bench)."""
    return [math.pi / 2, -math.pi / 2] if bench[3] > bench[4] else [0.0, math.pi]


def place_headings():
    return [k * math.pi / 4 for k in range(8)]
''')
print("[PATCH] goto_planner.py written")

# 3) check_scene.py: 1-second offline check of a scene (system python), before starting Isaac
open(f"{N}/check_scene.py", "w").write('''#!/usr/bin/env python3
"""Check the current scene (nav/scene.json or defaults) before running Isaac: pick and place approaches.
Usage: python3 check_scene.py        (NAV_SCENE=/path/to/file.json to check another file)"""
import math, sys
sys.path.insert(0, "@SPOT_DEMO_ROOT@/nav")
import nav_world as W
from goto_planner import GotoPlanner, pick_headings, place_headings
print(f"[SCENE] file: {W.SCENE_FILE} ({'used' if W.SCENE else 'not found -> defaults'})")
print(f"[SCENE] start {W.START[:2]} | bench {W.BENCH[1:5]} | bin {W.BIN[1:3]} | surprise {[b[:3] for b in W.SURPRISE]}")
gp = GotoPlanner(W.OBSTACLES)
d, who = gp.dist(*W.START[:2])
print(f"[SCENE] start: nearest obstacle {who} {d:.2f} m -> {'OK' if d >= gp.turn_r else 'TOO CLOSE to turn'}")
pk = gp.plan((W.BENCH[1], W.BENCH[2]), W.STANDOFF, pick_headings(W.BENCH), robot_xy=W.START[:2], label="pick")
for t in pk["tried"]:
    print("[SCENE]   pick  " + t)
pl = gp.plan((W.BIN[1], W.BIN[2]), 0.78, place_headings(), robot_xy=pk["stand"][:2] if pk["ok"] else W.START[:2], label="place")
for t in pl["tried"]:
    print("[SCENE]   place " + t)
ok = pk["ok"] and pl["ok"]
for r in (pk, pl):
    if r["ok"]:
        print(f"[SCENE] {r['reason']}: stand ({r['stand'][0]:+.2f},{r['stand'][1]:+.2f}) pre-pose ({r['pre'][0]:+.2f},{r['pre'][1]:+.2f})")
    else:
        print(f"[SCENE] UNREACHABLE - {r['reason']}")
print("[SCENE] RESULT:", "OK - run it" if ok else "NOT RUNNABLE - change the scene")
sys.exit(0 if ok else 1)
''')
print("[PATCH] check_scene.py written")

# 4) scene presets
os.makedirs(f"{N}/scenes", exist_ok=True)
presets = {
    "default": {},
    "bench_north_east": {"bench": [3.2, 2.3, "x"], "bin": [3.6, -2.9], "surprise": []},
    "bin_east": {"bin": [3.6, -2.9]},
}
for name, sc in presets.items():
    json.dump(sc, open(f"{N}/scenes/{name}.json", "w"), indent=1)
print(f"[PATCH] scenes written: {sorted(presets)}")

# 5) nav_up.sh: regenerate the map from the current scene on every start
p = f"{N}/nav_up.sh"
s = open(p).read()
if "gen_map.py" not in s:
    o = "trap 'echo \"[UP] stopping everything\"; kill 0' EXIT INT TERM\n"
    assert s.count(o) == 1, "nav_up.sh trap line not found"
    s = s.replace(o, o + "python3 $N/gen_map.py | grep -E \"MAP\\]\" || exit 1      # map always matches nav_world / scene.json\n")
    open(p, "w").write(s)
    print("[PATCH] nav_up.sh: regenerates the map on start")
else:
    print("[PATCH] nav_up.sh already regenerates the map")

# 6) demo: bench/bin from the scene, planned stand + pre-poses, all headings from the plan
src, dst = f"{C}/full_demo_perc_nav_amcl.py", f"{C}/full_demo_goto.py"
s = open(src).read()
R = [
 ('cfg.scene.table.init_state.pos = (BENCH_X + args.shift, 0.0, BENCH_H / 2)\ncfg.scene.object.init_state.pos = [BENCH_X + args.shift, 0.0, CUBE_Z]\n',
  'cfg.scene.table.init_state.pos = (W.BENCH[1], W.BENCH[2], BENCH_H / 2)          # GOTO: bench from nav_world / scene.json\n'
  'cfg.scene.object.init_state.pos = [W.BENCH[1], W.BENCH[2], CUBE_Z]\n'
  '_long_x = W.BENCH[3] > W.BENCH[4]\n'
  'if _long_x:                                   # bench turned 90 deg: swap its size instead of rotating it\n'
  '    _sz = tuple(cfg.scene.table.spawn.size)\n'
  '    cfg.scene.table.spawn.size = (_sz[1], _sz[0], _sz[2])\n'
  'print(f"[GOTO] bench at ({W.BENCH[1]:+.2f},{W.BENCH[2]:+.2f}) long side along {\'x\' if _long_x else \'y\'} | scene file {W.SCENE_FILE if W.SCENE else \'none (defaults)\'}", flush=True)\n'),
 ('"pose_range": {"x": (-0.03, 0.03), "y": (-0.10, 0.10), "z": (0.0, 0.0)}}\n',
  '"pose_range": ({"x": (-0.10, 0.10), "y": (-0.03, 0.03), "z": (0.0, 0.0)} if W.BENCH[3] > W.BENCH[4] else\n'
  '                                                         {"x": (-0.03, 0.03), "y": (-0.10, 0.10), "z": (0.0, 0.0)})}\n'),
 ('bench_park = origin + torch.tensor([args.shift, 0.0], device=dev)\n',
  'from goto_planner import GotoPlanner, pick_headings, place_headings\n'
  'gp = GotoPlanner(W.OBSTACLES, pre_dist=args.pre_dist)\n'
  '_pp = gp.plan((W.BENCH[1], W.BENCH[2]), W.STANDOFF, pick_headings(W.BENCH), robot_xy=START_XY, label="pick")\n'
  'for _t in _pp["tried"]:\n'
  '    print(f"[GOTO]   pick  {_t}", flush=True)\n'
  'if not _pp["ok"]:\n'
  '    print(f"[RESULT] NAV FAILED: {_pp[\'reason\']} - change the scene (check_scene.py)", flush=True)\n'
  '    env.close(); app.close(); sys.exit(1)\n'
  'PICK_YAW = _pp["stand"][2]\n'
  'bench_park = origin + torch.tensor(_pp["stand"][:2], device=dev)\n'),
 ('bin_park = bin_w + torch.tensor([0.0, args.place_dist], device=dev)\n',
  '_pl = gp.plan((BIN_X, BIN_Y), args.place_dist, place_headings(), robot_xy=_pp["stand"][:2], label="place")\n'
  'for _t in _pl["tried"]:\n'
  '    print(f"[GOTO]   place {_t}", flush=True)\n'
  'if not _pl["ok"]:\n'
  '    print(f"[RESULT] NAV FAILED: {_pl[\'reason\']} - change the scene (check_scene.py)", flush=True)\n'
  '    env.close(); app.close(); sys.exit(1)\n'
  'PLACE_YAW = _pl["stand"][2]\n'
  'bin_park = origin + torch.tensor(_pl["stand"][:2], device=dev)\n'
  'print(f"[GOTO] {_pp[\'reason\']} | {_pl[\'reason\']}", flush=True)\n'),
 ('prepick = bench_park - torch.tensor([args.pre_dist, 0.0], device=dev)\n'
  'preplace = bin_park + torch.tensor([0.0, args.pre_dist], device=dev)\n',
  'prepick = origin + torch.tensor(_pp["pre"][:2], device=dev)      # GOTO: checked pre-poses (room to turn)\n'
  'preplace = origin + torch.tensor(_pl["pre"][:2], device=dev)\n'),
 ('| pre-pick {rel(prepick)} yaw 0 | "\n',
  '| pre-pick {rel(prepick)} yaw {math.degrees(PICK_YAW):+.0f} | "\n'),
 ('pre-place {rel(preplace)} yaw -90 | bin park',
  'pre-place {rel(preplace)} yaw {math.degrees(PLACE_YAW):+.0f} | bin park'),
 ('                legs = nav_phase(prepick, 0.0, lambda: enter("APPROACH"), nav_cmd)\n',
  '                legs = nav_phase(prepick, PICK_YAW, lambda: enter("APPROACH"), nav_cmd)\n'),
 ('                a, legs = drive(bench_park, 0.0, stow, True, lambda: enter("UNFOLD", hold_xy=bench_park, hold_yaw=0.0))\n',
  '                a, legs = drive(bench_park, PICK_YAW, stow, True, lambda: enter("UNFOLD", hold_xy=bench_park, hold_yaw=PICK_YAW))\n'),
 ('                    S["bk"] = pp - torch.tensor([0.6, 0.0], device=dev)     # step back from the bench before Nav2 starts\n',
  '                    S["bk"] = prepick.clone()                              # GOTO: back off to the checked pre-pose\n'),
 ('                a, legs = drive(S["bk"], 0.0, stow, False, lambda: enter("NAV_BIN"), timeout=6.0)\n',
  '                a, legs = drive(S["bk"], PICK_YAW, stow, False, lambda: enter("NAV_BIN"), timeout=8.0)\n'),
 ('                legs = nav_phase(preplace, -math.pi / 2, _to_go_bin, nav_cmd)\n',
  '                legs = nav_phase(preplace, PLACE_YAW, _to_go_bin, nav_cmd)\n'),
 ('    state["wp"], state["wp_i"] = [(bin_park, -math.pi / 2)], 0\n',
  '    state["wp"], state["wp_i"] = [(bin_park, PLACE_YAW)], 0\n'),
 ('                        enter("PLACE", hold_xy=bin_park, hold_yaw=-math.pi / 2)\n',
  '                        enter("PLACE", hold_xy=bin_park, hold_yaw=PLACE_YAW)\n'),
]
for old, new in R:
    n = s.count(old)
    assert n == 1, f"[MERGE] FAILED: pattern found {n} times (need 1): {old[:80]!r}"
    s = s.replace(old, new)
open(dst, "w").write(s)
print(f"[MERGE] OK: {len(R)} changes -> {dst}")
