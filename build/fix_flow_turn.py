"""FLOW fix: after BACKOFF_BIN, turn in place with the scripted walk to face the table, THEN hand over to Nav2.
Reason: Nav2's rotate-in-place command is wz -0.30; with wz_bias +0.10 the walking policy gets only -0.20 rad/s and
stalls (yaw frozen, tilt 5.9 deg, 60 s lost in round 3). The scripted turn uses up to 0.6 rad/s.
Applied after fix_flow_backoff.py. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if 'ph == "TURN_TABLE"' in s:
    print("[FIX] TURN_TABLE already there"); sys.exit(0)
o1 = '                a, legs = drive(preplace, PLACE_YAW, stow, True, lambda: enter("NAV_TABLE"), timeout=6.0)\n'
o2 = '            elif ph == "NAV_TABLE":                                  # FLOW: back from the bin to the table view pose\n'
for o in (o1, o2):
    assert s.count(o) == 1, f"[FIX] FAILED: pattern found {s.count(o)} times: {o[:70]!r} (run fix_flow_backoff.py first)"
s = s.replace(o1, '                a, legs = drive(preplace, PLACE_YAW, stow, True, lambda: enter("TURN_TABLE"), timeout=6.0)\n')
s = s.replace(o2,
  '            elif ph == "TURN_TABLE":                                 # FLOW: scripted turn to face the table before Nav2\n'
  '                _d = VIEW[0] - preplace\n'
  '                a, legs = drive(preplace, math.atan2(float(_d[1]), float(_d[0])), stow, True, lambda: enter("NAV_TABLE"), timeout=8.0)\n'
  + o2)
open(p, "w").write(s)
print("[FIX] TURN_TABLE added: bin -> back 0.5 m -> turn to face the table (scripted) -> Nav2")
