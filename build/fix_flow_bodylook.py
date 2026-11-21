"""FLOW + BODY CAMERA: take the body-camera look from the table PRE-POSE (0.5 m back), not the stand pose.
At the stand pose the camera is ~10 cm from the bench edge and the stowed gripper hangs in front of it; from the pre-pose
it sees the whole bench and all cubes (checked in the GUI: translate 0.47/0/0, 20 deg down). So with --body_cam, when no cube
is chosen yet, Spot parks at the pre-pose, looks, asks, and then approaches the chosen cube. Needs fix_flow_bodycam.py
and fix_flow_turnshift.py. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if "BODY CAM: look from the pre-pose" in s:
    print("[FIX] body camera look from the pre-pose already there"); sys.exit(0)
if "BODY CAM:" not in s:
    print("[FIX] body look: needs fix_flow_bodycam.py first - skipped"); sys.exit(0)
R = [
 ('                a, legs = drive(bench_park, PICK_YAW, stow, True, _after_approach)\n',
  '                if args.body_cam and args.multi_flow and SEL["colour"] is None:      # BODY CAM: look from the pre-pose\n'
  '                    a, legs = drive(prepick, PICK_YAW, stow, True,\n'
  '                                    lambda: (DET.clear(), FLOW.update(scan_i=0), enter("BODYLOOK", hold_xy=prepick, hold_yaw=PICK_YAW)))\n'
  '                else:\n'
  '                    a, legs = drive(bench_park, PICK_YAW, stow, True, _after_approach)\n'),
 ('                _d = prepick - VIEW[0]\n'
  '                _wy = math.atan2(float(_d[1]), float(_d[0]))\n',
  '                _d = prepick - VIEW[0]\n'
  '                _wy = math.atan2(float(_d[1]), float(_d[0])) if float(torch.norm(_d)) > 0.05 else PICK_YAW   # same pre-pose: no turn\n'),
]
for old, new in R:
    n = s.count(old)
    assert n == 1, f"[FIX] FAILED: pattern found {n} times (need 1): {old[:70]!r}"
    s = s.replace(old, new)
open(p, "w").write(s)
print("[FIX] body camera looks from the table pre-pose, then Spot approaches the chosen cube")
