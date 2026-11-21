"""Add the moving box to the lidar scan from its live pose (GPU PhysX ray casts do not see bodies moved by code).
Patches full_demo_mover.py in place; safe to run twice. Run again after any rebuild with make_mover_patch.py."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_mover.py"
s = open(p).read()
if "_scan_mover" in s:
    print("[FIX] already applied"); raise SystemExit
o = "if args.mover_test:\n"
assert s.count(o) == 1, "mover_test line not found"
s = s.replace(o,
    'if mover is not None:                         # moving box -> scan from its live pose (ray casts miss moved bodies on GPU)\n'
    '    _ms = Scan2D([("mover", 0.0, 0.0, W.MOVER["size"][0], W.MOVER["size"][1], 0.0, True)], n_rays=args.scan_rays)\n'
    '    _base_scan = link.scan\n'
    '    def _scan_mover(x, y, yaw):\n'
    '        sx, sy = W.MOVER["size"][0], W.MOVER["size"][1]\n'
    '        mx, my = MV["xy"]\n'
    '        _ms.boxes = np.array([[mx - sx / 2, mx + sx / 2, my - sy / 2, my + sy / 2]])\n'
    '        return np.minimum(_base_scan(x, y, yaw), _ms(x, y, yaw))   # nearest hit wins (box also hides what is behind it)\n'
    '    link.scan = _scan_mover\n'
    '    print("[MOVER] scan = PhysX ray casts (static scene) + moving box from its live pose", flush=True)\n'
    + o, 1)
a = '        if args.scan == "physx":\n            pr = _scan.hit_paths\n'
b = '    print("[MOVER] test done - quitting", flush=True)\n'
i, j = s.index(a), s.index(b)
s = s[:i] + ('        lx, ly = float(MV["xy"][0]) + 2.0, float(MV["xy"][1])\n'
             '        r = link.scan(lx, ly, math.pi)                     # look west at the box from 2 m east\n'
             '        ahead = float(r[len(r) // 2])                      # ray at angle 0 = straight ahead\n'
             '        print(f"[MOVER] test: lidar from 2 m east, straight ahead {ahead:.2f} m (box face expected at {2.0 - W.MOVER[\'size\'][0] / 2:.2f} m) "\n'
             '              f"-> {\'SEES THE BOX\' if abs(ahead - (2.0 - W.MOVER[\'size\'][0] / 2)) < 0.05 else \'DOES NOT SEE THE BOX\'}", flush=True)\n') + s[j:]
open(p, "w").write(s)
print("[FIX] moving box added to the scan; mover_test now checks the range straight at the box")
