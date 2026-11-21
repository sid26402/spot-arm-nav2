"""FLOW: --stand_extra (m) plans each chosen cube's stand pose this much further back from the cube.
Reason (v8.18 body-camera run): the cube ended at x 0.83-0.84 in the base frame (wanted 0.875; perception's misses are at
0.78-0.85, good grasps 0.88-0.90). Systematic: body camera ~1 cm short + settle-hold residual ~1.7 cm + approach ~1 cm.
Default 0.03 (measured: puts the cube at x 0.88). Applied after fix_flow_bodycam.py. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if "--stand_extra" in s:
    print("[FIX] --stand_extra already there"); sys.exit(0)
R = [
 ('parser.add_argument("--order"',
  'parser.add_argument("--stand_extra", type=float, default=0.03, help="FLOW: plan the cube stand pose this much further back (m)")\n'
  'parser.add_argument("--order"'),
 ('    r = gp.plan(xy, W.STANDOFF, [PICK_YAW], label=f"pick {colour}")\n',
  '    r = gp.plan(xy, W.STANDOFF + args.stand_extra, [PICK_YAW], label=f"pick {colour}")   # STANDOFF: --stand_extra\n'),
]
for old, new in R:
    assert s.count(old) == 1, f"[FIX] FAILED: pattern found {s.count(old)} times: {old[:60]!r}"
    s = s.replace(old, new)
open(p, "w").write(s)
print("[FIX] --stand_extra: cube stand pose planned further back (default 0)")
