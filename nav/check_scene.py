#!/usr/bin/env python3
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
