"""FLOW: no nav yaw bias during PICK. Perception v8.22 commands the legs exactly [0, 0, 0] while the arm policy picks;
the nav build added --wz_bias (+0.10) to every non-Nav2 phase, PICK included, so the legs got [0, 0, 0.10].
This adds "PICK" to the phases that skip the bias. Every other scripted phase keeps it. Safe to run twice."""
import re, sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
pat = re.compile(r'( +if ph not in \()([^)]*)(\) and not args\.no_script_bias:\n)')
m = pat.findall(s)
assert len(m) == 1, f"[FIX] FAILED: bias line found {len(m)} times"
if '"PICK"' in m[0][1]:
    print("[FIX] PICK already skips the yaw bias"); sys.exit(0)
s = pat.sub(lambda g: g.group(1) + g.group(2) + ', "PICK"' + g.group(3), s)
open(p, "w").write(s)
print("[FIX] PICK: legs get exactly the perception command (no +wz_bias)")
