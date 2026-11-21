"""FLOW log for perception: at every LOOK, print the selected cube's yaw from sim truth (LOG ONLY, never used to decide).
A cube is 90-deg symmetric, so 'off' = yaw folded into -45..+45 (0 = square to the bench, as in training). Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if "cube yaw (sim truth, log only)" in s:
    print("[FIX] cube yaw log already there"); sys.exit(0)
o = '    print(f"[DEMO] {p}  (sim {globals().get(\'step\', 0) * dt:.1f}s | wall {time.time() - T_START:.1f}s)", flush=True)\n'
assert s.count(o) == 1, f"[FIX] FAILED: enter() print line found {s.count(o)} times"
s = s.replace(o, o +
  '    if p == "LOOK" and args.multi_flow:\n'
  '        try:\n'
  '            _w, _x, _y, _z = [float(v) for v in obj.data.root_quat_w[0]]\n'
  '            _yaw = math.degrees(math.atan2(2 * (_w * _z + _x * _y), 1 - 2 * (_y * _y + _z * _z)))\n'
  '            _tl = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (_x * _x + _y * _y)))))\n'
  '            print(f"[MULTI] {SEL[\'colour\']} cube yaw (sim truth, log only): {_yaw:+.1f} deg | off square {((_yaw + 45) % 90) - 45:+.1f} deg | tilt {_tl:.1f} deg", flush=True)\n'
  '        except Exception as _e:\n'
  '            print(f"[MULTI] cube yaw log failed: {_e}", flush=True)\n')
open(p, "w").write(s)
print("[FIX] cube yaw log at every LOOK (sim truth, log only)")
