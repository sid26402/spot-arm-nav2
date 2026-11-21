"""FLOW: stand 10 cm further from the bin when placing (--place_dist default 0.78 -> 0.88). In six places at 0.78 the
release point was 8-18 cm past the bin centre; the PLACE_FIX backward step works at the original bin but barely moves at
bin2 (walking policy ignores small backward steps). At 0.88 bin2 needed only -1.6 cm. Override with --place_dist. Safe to run twice."""
import re, sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
pat = re.compile(r'(parser\.add_argument\("--place_dist", type=float, default=)([0-9.]+)(,)')
m = pat.findall(s)
assert len(m) == 1, f"[FIX] FAILED: --place_dist found {len(m)} times"
if m[0][1] == "0.88":
    print("[FIX] --place_dist already 0.88"); sys.exit(0)
s = pat.sub(lambda g: g.group(1) + "0.88" + g.group(3), s)
open(p, "w").write(s)
print(f"[FIX] --place_dist default {m[0][1]} -> 0.88 (stand further back at the bins)")
