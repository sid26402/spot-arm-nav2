"""FLOW fix: if the chosen cube's pre-pose is the table pre-pose (within 5 cm, e.g. the middle cube), skip the
turn-walk-turn and go straight to APPROACH. In the first body-camera run the dex cube's pre-pose was 2 cm from the table
pre-pose; SHIFT_WALK then tried a 9 cm backwards step, which the walking policy does not do (timeout, -9.2 cm).
Applied after fix_flow_bodylook.py. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if "same pre-pose: no shift needed" in s:
    print("[FIX] no-shift for the middle cube already there"); sys.exit(0)
o = '                if ph == "SHIFT":\n'
assert s.count(o) == 1, f"[FIX] FAILED: SHIFT branch found {s.count(o)} times"
s = s.replace(o, '                if float(torch.norm(_d)) <= 0.05:                    # same pre-pose: no shift needed, go straight in\n'
                 '                    a, legs = arm_action(stow, True), [0.0, 0.0, 0.0]\n'
                 '                    enter("APPROACH")\n'
                 '                elif ph == "SHIFT":\n')
open(p, "w").write(s)
print("[FIX] middle cube: no turn-walk-turn when the pre-pose is the same")
