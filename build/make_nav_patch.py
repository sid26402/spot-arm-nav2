"""Add a real-scene scan (PhysX ray casts) + a surprise obstacle to the nav stack. Idempotent checks, asserts on every change."""
import sys
N = "@SPOT_DEMO_ROOT@/nav" if len(sys.argv) < 2 else sys.argv[1]
# 1) nav_link.StateLink: optional scan function + keep the last scan
p = f"{N}/nav_link.py"
s = open(p).read()
if "scan_fn" not in s:
    for old, new in [
        ("    def __init__(self, obstacles, dt, rate=25.0, watchdog_s=0.5, lim=(1.0, 0.5, 1.0)):\n        self.dt, self.lim = dt, lim\n        self.scan = Scan2D(obstacles)\n",
         "    def __init__(self, obstacles, dt, rate=25.0, watchdog_s=0.5, lim=(1.0, 0.5, 1.0), scan_fn=None):\n        self.dt, self.lim = dt, lim\n        self.scan = Scan2D(obstacles) if scan_fn is None else scan_fn   # scan_fn(x, y, yaw) -> ranges, e.g. PhysxScan\n        self.last = None\n"),
        ("            r = self.scan(x, y, yaw)\n", "            r = self.scan(x, y, yaw)\n            self.last = r\n"),
    ]:
        assert s.count(old) == 1, "nav_link.py pattern mismatch: " + old[:60]
        s = s.replace(old, new)
    open(p, "w").write(s)
    print("[PATCH] nav_link.py: StateLink(scan_fn=...) added")
else:
    print("[PATCH] nav_link.py already patched")
# 2) nav_world: surprise obstacle (spawned in the sim, NOT in the Nav2 map)
p = f"{N}/nav_world.py"
s = open(p).read()
if "SURPRISE" not in s:
    s += ('\n# Surprise obstacle: spawned in the sim but NOT in OBSTACLES, so it is NOT in the Nav2 map.\n'
          '# Nav2 can only avoid it through the scan (PhysX lidar). Placed on the leg-1 route east of baffle_n3.\n'
          'SURPRISE = [("surprise_a", 0.85, 1.3, 0.4, 0.4, 0.6, True)]\n')
    open(p, "w").write(s)
    print("[PATCH] nav_world.py: SURPRISE added")
else:
    print("[PATCH] nav_world.py already has SURPRISE")
