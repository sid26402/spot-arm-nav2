"""FLOW fix: during SHIFT_WALK stop within 2 cm ALONG the walk direction (not the general 8 cm parking tolerance).
After turn-walk-turn the leftover error lies along the walk = sideways to the table, and the walking policy cannot
correct small sideways errors later. Forward/back corrections it does well, so finish them here. Prints the result.
Applied after fix_flow_turnshift.py. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if "shift walk done" in s:
    print("[FIX] precise shift walk already there"); sys.exit(0)
o = ('                else:\n'
     '                    a, legs = drive(prepick, _wy, stow, True, lambda: enter("SHIFT_BACK"), timeout=10.0)\n')
assert s.count(o) == 1, f"[FIX] FAILED: SHIFT_WALK branch found {s.count(o)} times (run fix_flow_turnshift.py first)"
n = ('                else:                                            # walk straight; stop within 2 cm ALONG the walk\n'
     '                    _p, _yw = base_pose()\n'
     '                    _e = prepick - _p\n'
     '                    _al = float(_e[0]) * math.cos(_wy) + float(_e[1]) * math.sin(_wy)\n'
     '                    _dy = wrap(_wy - _yw)\n'
     '                    _wz = max(-0.6, min(0.6, 1.5 * _dy)) if abs(_dy) > YAW_TOL * 0.5 else 0.0\n'
     '                    _vx = 0.0 if abs(_al) < 0.015 else math.copysign(max(0.15, min(0.4, 1.2 * abs(_al))), _al)\n'
     '                    legs = [_vx, 0.0, _wz]\n'
     '                    a = arm_action(stow, True)\n'
     '                    S["info"] = f"walk along={_al:+.3f} dyaw={math.degrees(_dy):+.0f}"\n'
     '                    S["good"] = S["good"] + 1 if abs(_al) < 0.02 and abs(_dy) < YAW_TOL else 0\n'
     '                    if S["good"] >= HOLD or t * dt > 10.0:\n'
     '                        _cr = -float(_e[0]) * math.sin(_wy) + float(_e[1]) * math.cos(_wy)\n'
     '                        print(f"[MULTI] shift walk done: {_al * 100:+.1f} cm along, {_cr * 100:+.1f} cm across"\n'
     '                              f"{\' (timeout)\' if S[\'good\'] < HOLD else \'\'}", flush=True)\n'
     '                        legs = [0.0, 0.0, 0.0]\n'
     '                        enter("SHIFT_BACK")\n')
s = s.replace(o, n)
open(p, "w").write(s)
print("[FIX] precise shift walk: stops within 2 cm along the walk direction")
