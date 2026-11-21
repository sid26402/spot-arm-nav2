"""FLOW fix: approach every chosen cube HEAD-ON instead of sidestepping along the bench, plus a parking gate.
Reason: the red sidestep slid along the bench front with +2 cm clearance; a 3 deg yaw slip put Spot's front corner
into the bench (tilt 2.5 deg, frozen 19 cm short) -> bad LOOK -> sideways pick -> arm jammed -> cube lost.
New: ASK pose -> BACK_TABLE (0.5 m straight back) -> SHIFT (sidestep at the pre-pose, 0.5 m from the bench) ->
APPROACH (straight in) -> gate: more than 9 cm (parking tolerance + 1 cm) off the stand pose -> REAPPROACH (back to pre-pose, in again), max 2.
Applied after make_flow_patch.py and fix_flow_backoff.py. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if 'ph == "BACK_TABLE"' in s:
    print("[FIX] head-on approach already there"); sys.exit(0)
R = [
 ('    enter("SHIFT" if far > 0.05 else "UNFOLD", hold_xy=bench_park, hold_yaw=PICK_YAW)\n',
  '    FLOW["park"] = 0\n'
  '    enter("BACK_TABLE" if far > 0.05 else "UNFOLD", hold_xy=bench_park, hold_yaw=PICK_YAW)\n'),
 ('def _after_approach():\n'
  '    if args.multi_flow and SEL["colour"] is None:\n',
  'def _after_approach():\n'
  '    if args.multi_flow:\n'
  '        pp, _ = base_pose()\n'
  '        err = float(torch.norm(bench_park - pp))\n'
  '        if err > POS_TOL + 0.01 and FLOW.get("park", 0) < 2:\n'
  '            FLOW["park"] = FLOW.get("park", 0) + 1\n'
  '            print(f"[MULTI] parked {err * 100:.0f} cm off the stand pose - back to the pre-pose and in again ({FLOW[\'park\']}/2)", flush=True)\n'
  '            enter("REAPPROACH"); return\n'
  '        if err > POS_TOL + 0.01:\n'
  '            print(f"[MULTI] WARNING: still {err * 100:.0f} cm off after 2 re-approaches - continuing, perception look poses must cope", flush=True)\n'
  '        FLOW["park"] = 0\n'
  '    if args.multi_flow and SEL["colour"] is None:\n'),
 ('    SEL.update(colour=None)\n',
  '    SEL.update(colour=None); FLOW["park"] = 0\n'),
 ('            elif ph == "SHIFT":                                      # FLOW: sidestep to stand in front of the chosen cube\n'
  '                a, legs = drive(bench_park, PICK_YAW, stow, True, lambda: enter("UNFOLD", hold_xy=bench_park, hold_yaw=PICK_YAW), timeout=10.0)\n',
  '            elif ph == "BACK_TABLE":                                 # FLOW: straight back 0.5 m from the bench (clearance to sidestep)\n'
  '                a, legs = drive(VIEW[0], PICK_YAW, stow, True, lambda: enter("SHIFT"), timeout=6.0)\n'
  '            elif ph == "SHIFT":                                      # FLOW: sidestep at the pre-pose, in front of the chosen cube\n'
  '                a, legs = drive(prepick, PICK_YAW, stow, True, lambda: enter("APPROACH"), timeout=10.0)\n'
  '            elif ph == "REAPPROACH":                                 # FLOW: parked badly - back to the pre-pose, then APPROACH again\n'
  '                a, legs = drive(prepick, PICK_YAW, stow, True, lambda: enter("APPROACH"), timeout=8.0)\n'),
]
for old, new in R:
    n = s.count(old)
    assert n == 1, f"[FIX] FAILED: pattern found {n} times (need 1): {old[:70]!r}"
    s = s.replace(old, new)
open(p, "w").write(s)
print("[FIX] head-on approach: BACK_TABLE -> SHIFT at pre-pose -> APPROACH, parking gate 9 cm (2 re-approaches)")
