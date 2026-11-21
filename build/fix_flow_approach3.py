"""FLOW fix: make the approach's minimum turn a NET turn. The +wz_bias (0.10) is added to every scripted leg command later,
so a -0.25 rad/s correction became only -0.15 net - and clockwise is exactly the direction our yaw errors (+1..+4 deg)
need; net -0.20 already stalled the policy before. Now the command is net +-0.30 rad/s minimum (bias pre-subtracted).
Applied after fix_flow_approach2.py. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if "NET turn" in s:
    print("[FIX] net approach turn already there"); sys.exit(0)
o = '                    _wz = 0.0 if abs(_dy) < 0.015 else math.copysign(max(0.25, min(0.6, 1.5 * abs(_dy))), _dy)\n'
assert s.count(o) == 1, f"[FIX] FAILED: approach turn line found {s.count(o)} times (run fix_flow_approach2.py first)"
s = s.replace(o, '                    _wz = 0.0 if abs(_dy) < 0.015 else (math.copysign(max(0.30, min(0.6, 1.5 * abs(_dy))), _dy)\n'
                 '                                                       - (0.0 if args.no_script_bias else args.wz_bias))   # NET turn >= 0.30 after the bias\n')
open(p, "w").write(s)
print("[FIX] approach: minimum NET turn 0.30 rad/s in both directions (bias pre-subtracted)")
