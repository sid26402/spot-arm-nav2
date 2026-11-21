"""FLOW fix: precise final APPROACH. The general parking controller stops anywhere inside 8 cm and slows to 0.15 m/s, which
the walking policy sometimes ignores: dex APPROACH stalled 12 cm short for 12 s, then parked 5.5 cm short (inside the 8 cm
tolerance), and the wrist camera never saw the cube (its view is only ~+-10 cm deep at the bench). New: walk straight in
along the pick heading at >= 0.2 m/s, stop within 3 cm ALONG the heading; sideways only if more than 4 cm off.
Prints the result. Applied after fix_flow_bodylook.py. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if "approach done:" in s:
    print("[FIX] precise approach already there"); sys.exit(0)
o = '                    a, legs = drive(bench_park, PICK_YAW, stow, True, _after_approach)\n'
assert s.count(o) == 1, f"[FIX] FAILED: APPROACH line found {s.count(o)} times (run fix_flow_bodylook.py first)"
n = ('                    _p, _yw = base_pose()                              # straight in, stop within 3 cm along the heading\n'
     '                    _e = bench_park - _p\n'
     '                    _al = float(_e[0]) * math.cos(PICK_YAW) + float(_e[1]) * math.sin(PICK_YAW)\n'
     '                    _cr = -float(_e[0]) * math.sin(PICK_YAW) + float(_e[1]) * math.cos(PICK_YAW)\n'
     '                    _dy = wrap(PICK_YAW - _yw)\n'
     '                    _wz = max(-0.6, min(0.6, 1.5 * _dy)) if abs(_dy) > YAW_TOL * 0.5 else 0.0\n'
     '                    _vx = 0.0 if abs(_al) < 0.025 else math.copysign(max(0.2, min(0.4, 1.2 * abs(_al))), _al)\n'
     '                    _vy = 0.0 if abs(_cr) < 0.04 else math.copysign(0.2, _cr)\n'
     '                    legs = [_vx, _vy, _wz]\n'
     '                    a = arm_action(stow, True)\n'
     '                    S["info"] = f"approach along={_al:+.3f} across={_cr:+.3f} dyaw={math.degrees(_dy):+.0f}"\n'
     '                    S["good"] = S["good"] + 1 if abs(_al) < 0.03 and abs(_dy) < YAW_TOL else 0\n'
     '                    if S["good"] >= HOLD or t * dt > 15.0:\n'
     '                        print(f"[MULTI] approach done: {_al * 100:+.1f} cm along, {_cr * 100:+.1f} cm across"\n'
     '                              f"{\' (timeout)\' if S[\'good\'] < HOLD else \'\'}", flush=True)\n'
     '                        legs = [0.0, 0.0, 0.0]\n'
     '                        _after_approach()\n')
s = s.replace(o, n)
open(p, "w").write(s)
print("[FIX] precise approach: straight in, within 3 cm along the pick heading")
