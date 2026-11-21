"""FLOW fix: after a cube is binned, step 0.5 m back to the pre-place pose (scripted) before Nav2 walks back to the table.
Reason: the bin stand pose has -1 cm clearance, so Nav2 started inside the inflated wall -> controller aborts, recovery
spins, and in round 3 Spot wedged against the wall (tilt 5.8 deg, no motion for 90 s). Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if 'ph == "BACKOFF_BIN"' in s:
    print("[FIX] BACKOFF_BIN already there"); sys.exit(0)
o1 = '    _say(f"back to the table - left: {REMAIN}")\n    enter("NAV_TABLE")\n'
o2 = '            elif ph == "NAV_TABLE":                                  # FLOW: back from the bin to the table view pose\n'
o3 = '            if ph not in ("NAV_PICK", "NAV_BIN", "NAV_FREE", "NAV_TABLE") and not args.no_script_bias:\n'
for o in (o1, o2, o3):
    assert s.count(o) == 1, f"[FIX] FAILED: pattern found {s.count(o)} times: {o[:70]!r}"
s = s.replace(o1, '    _say(f"back to the table - left: {REMAIN}")\n    enter("BACKOFF_BIN")\n')
s = s.replace(o2, '            elif ph == "BACKOFF_BIN":                                # FLOW: step back from the bin to the pre-place pose\n'
                  '                a, legs = drive(preplace, PLACE_YAW, stow, True, lambda: enter("NAV_TABLE"), timeout=6.0)\n' + o2)
open(p, "w").write(s)
print("[FIX] BACKOFF_BIN added: bin -> pre-place (0.5 m back, scripted) -> Nav2 to the table")
