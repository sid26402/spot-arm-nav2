"""Small nav fixes on the final demo file. Safe to run twice.
 - BACKOFF: Nav2 takes over right after, so give up after 4 s instead of 8 s (it often stopped 8.0 cm out, on the 8 cm edge)."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_multi.py"
s = open(p).read()
o = 'lambda: enter("NAV_BIN"), timeout=8.0)'
if o in s:
    open(p, "w").write(s.replace(o, 'lambda: enter("NAV_BIN"), timeout=4.0)', 1)); print("[FIX] BACKOFF timeout 8 s -> 4 s")
else:
    print("[FIX] BACKOFF timeout already 4 s" if 'lambda: enter("NAV_BIN"), timeout=4.0)' in s else "[FIX] WARNING: BACKOFF line not found")
