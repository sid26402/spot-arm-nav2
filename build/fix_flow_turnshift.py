"""FLOW fix: no pure sidesteps. The walking policy does not do small sideways steps (seen by nav AND perception:
SHIFT 0.30 m stalled 12 cm short in 2 of 3 tries). New: at the table pre-pose turn to face the chosen cube's
pre-pose -> walk forward to it -> turn back to the table -> APPROACH. Applied after fix_flow_headon.py. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if 'enter("SHIFT_WALK")' in s:
    print("[FIX] turn-walk-turn shift already there"); sys.exit(0)
o = ('            elif ph == "SHIFT":                                      # FLOW: sidestep at the pre-pose, in front of the chosen cube\n'
     '                a, legs = drive(prepick, PICK_YAW, stow, True, lambda: enter("APPROACH"), timeout=10.0)\n')
assert s.count(o) == 1, f"[FIX] FAILED: SHIFT phase found {s.count(o)} times (run fix_flow_headon.py first)"
n = ('            elif ph in ("SHIFT", "SHIFT_WALK"):                       # FLOW: turn to the new pre-pose, then walk forward to it\n'
     '                _d = prepick - VIEW[0]\n'
     '                _wy = math.atan2(float(_d[1]), float(_d[0]))\n'
     '                if ph == "SHIFT":\n'
     '                    a, legs = drive(VIEW[0], _wy, stow, True, lambda: enter("SHIFT_WALK"), timeout=8.0)\n'
     '                else:\n'
     '                    a, legs = drive(prepick, _wy, stow, True, lambda: enter("SHIFT_BACK"), timeout=10.0)\n'
     '            elif ph == "SHIFT_BACK":                                 # FLOW: turn back to face the table\n'
     '                a, legs = drive(prepick, PICK_YAW, stow, True, lambda: enter("APPROACH"), timeout=8.0)\n')
s = s.replace(o, n)
open(p, "w").write(s)
print("[FIX] shift = turn -> walk forward -> turn back (no sidesteps)")
