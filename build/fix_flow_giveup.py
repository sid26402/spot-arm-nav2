"""FLOW + perception v8: what happens when perception gives up on a cube (rotated/tipped/false grasp/no retries/not seen).
Perception calls give_up() -> GIVEUP (arm to stow, base held) -> GIVEUP_DONE -> after_giveup(). In --multi_flow this file's
after_giveup() wins (defined later in the file): drop that cube, back to the ASK pose, ask about the rest; if the cube is
still held after a failed PLACE -> stop the run with a report. Also resets perception's per-cube counters when a new cube
is chosen. Needs perception v8 (give_up / after_giveup); with v6 it does nothing. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if "FLOW after_giveup" in s:
    print("[FIX] flow give-up handling already there"); sys.exit(0)
if "def after_giveup" not in s:
    print("[FIX] flow give-up: perception file has no after_giveup() (v6) - skipped"); sys.exit(0)
R = [
 ('    P["tries"] = 0; state["place_tries"] = 0\n',
  '    P["tries"] = 0; state["place_tries"] = 0\n'
  '    P["look_i"] = 0; P["reparks"] = 0; state["giveup_hold"] = False      # FLOW: fresh perception counters per cube\n'),
 ("        print(f\"[MULTI] all cubes done: {FLOW['placed']}\", flush=True); _say(\"all cubes done\"); return\n",
  "        print(f\"[MULTI] all cubes done: placed {FLOW['placed']} | skipped/failed {FLOW.get('failed', [])}\", flush=True); _say(\"all cubes done\"); return\n"),
 ('# ---------------- PANEL: operator panel (nav side) ----------------\n',
  'if args.multi_flow:\n'
  '    def after_giveup():                        # FLOW after_giveup: replaces perception\'s default (end the run)\n'
  '        global bench_park, prepick\n'
  '        c = SEL["colour"]\n'
  '        FLOW.setdefault("failed", [])\n'
  '        if state.get("giveup_hold"):\n'
  '            print(f"[MULTI] stopping: the {c} cube is still in the gripper after a failed PLACE | placed {FLOW[\'placed\']} | skipped/failed {FLOW[\'failed\'] + [c]}", flush=True)\n'
  '            enter("DONE"); return\n'
  '        if c in REMAIN:\n'
  '            REMAIN.remove(c); FLOW["failed"].append(c)\n'
  '        if not REMAIN:\n'
  '            print(f"[MULTI] all cubes done: placed {FLOW[\'placed\']} | skipped/failed {FLOW[\'failed\']}", flush=True)\n'
  '            enter("DONE"); return\n'
  '        SEL.update(colour=None); FLOW["park"] = 0\n'
  '        prepick, bench_park = VIEW[0].clone(), VIEW[1].clone()\n'
  '        print(f"[MULTI] gave up on the {c} cube - back to the table, left: {REMAIN}", flush=True)\n'
  '        enter("REAPPROACH")\n'
  '# ---------------- PANEL: operator panel (nav side) ----------------\n'),
]
for old, new in R:
    n = s.count(old)
    assert n == 1, f"[FIX] FAILED: pattern found {n} times (need 1): {old[:70]!r}"
    s = s.replace(old, new)
open(p, "w").write(s)
print("[FIX] flow give-up handling: skip the cube and ask about the rest (stop if a cube is still held)")
