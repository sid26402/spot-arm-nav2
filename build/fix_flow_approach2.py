"""FLOW fix: approach heading. Data from the body-camera run (base at LOOK vs cube): dex 4.5 cm left, yaw +1; red yaw +3;
blue 5 cm left AND yaw +4 -> cube 10 cm to the side in the wrist camera. Two causes: small sideways errors are never
corrected (the policy ignores small sideways steps) and small yaw commands (< ~0.1 rad/s after the +0.10 bias) do nothing.
New final approach: while more than 10 cm out, steer TOWARD the stand pose (turns the sideways error into heading, which
the policy follows); then turn to the pick heading with a minimum 0.25 rad/s yaw command; arrive only within 3 cm along
and 1.7 deg. Prints the final heading too. Applied after fix_flow_approach.py. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if "steer toward the stand pose" in s:
    print("[FIX] approach heading already there"); sys.exit(0)
R = [
 ('                    _dy = wrap(PICK_YAW - _yw)\n'
  '                    _wz = max(-0.6, min(0.6, 1.5 * _dy)) if abs(_dy) > YAW_TOL * 0.5 else 0.0\n'
  '                    _vx = 0.0 if abs(_al) < 0.025 else math.copysign(max(0.2, min(0.4, 1.2 * abs(_al))), _al)\n'
  '                    _vy = 0.0 if abs(_cr) < 0.04 else math.copysign(0.2, _cr)\n',
  '                    _hd = PICK_YAW\n'
  '                    if _al > 0.10 and abs(_cr) > 0.015:               # steer toward the stand pose while far out\n'
  '                        _hd = PICK_YAW + max(-0.25, min(0.25, math.atan2(_cr, _al)))\n'
  '                    _dy = wrap(_hd - _yw)\n'
  '                    _wz = 0.0 if abs(_dy) < 0.015 else math.copysign(max(0.25, min(0.6, 1.5 * abs(_dy))), _dy)\n'
  '                    _vx = 0.0 if abs(_al) < 0.025 else math.copysign(max(0.2, min(0.4, 1.2 * abs(_al))), _al)\n'
  '                    _vy = 0.0\n'),
 ('                    S["good"] = S["good"] + 1 if abs(_al) < 0.03 and abs(_dy) < YAW_TOL else 0\n',
  '                    _yf = wrap(PICK_YAW - _yw)\n'
  '                    S["good"] = S["good"] + 1 if abs(_al) < 0.03 and abs(_yf) < 0.03 else 0\n'),
 ('                        print(f"[MULTI] approach done: {_al * 100:+.1f} cm along, {_cr * 100:+.1f} cm across"\n',
  '                        print(f"[MULTI] approach done: {_al * 100:+.1f} cm along, {_cr * 100:+.1f} cm across, heading {math.degrees(_yf):+.1f} deg"\n'),
]
for old, new in R:
    n = s.count(old)
    assert n == 1, f"[FIX] FAILED: pattern found {n} times (need 1): {old[:70]!r} (run fix_flow_approach.py first)"
    s = s.replace(old, new)
open(p, "w").write(s)
print("[FIX] approach heading: steer toward the stand pose, final heading within 1.7 deg (min 0.25 rad/s turn)")
