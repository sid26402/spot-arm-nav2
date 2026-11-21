#!/usr/bin/env python3
"""Rasterise nav_world.OBSTACLES into a Nav2 map (PGM + YAML, 0.05 m). Pure python, no deps."""
import math, os, sys
sys.path.insert(0, "@SPOT_DEMO_ROOT@/nav")
import nav_world as W

RES, MARGIN = 0.05, 0.5
X0, X1 = W.ROOM_X[0] - MARGIN, W.ROOM_X[1] + MARGIN
Y0, Y1 = W.ROOM_Y[0] - MARGIN, W.ROOM_Y[1] + MARGIN
w, h = int(math.ceil((X1 - X0) / RES)), int(math.ceil((Y1 - Y0) / RES))
FREE, OCC = 254, 0
grid = bytearray([FREE]) * (w * h)
for name, cx, cy, sx, sy, _, _ in W.OBSTACLES:
    i0 = max(0, int(math.floor((cx - sx / 2 - X0) / RES)))
    i1 = min(w, int(math.ceil((cx + sx / 2 - X0) / RES)))
    j0 = max(0, int(math.floor((cy - sy / 2 - Y0) / RES)))
    j1 = min(h, int(math.ceil((cy + sy / 2 - Y0) / RES)))
    for j in range(j0, j1):
        row = h - 1 - j                      # PGM row 0 = top = max y
        for i in range(i0, i1):
            grid[row * w + i] = OCC
    print(f"  {name:7s} x[{cx - sx / 2:+.2f},{cx + sx / 2:+.2f}] y[{cy - sy / 2:+.2f},{cy + sy / 2:+.2f}]")
out = "@SPOT_DEMO_ROOT@/nav/maps"
os.makedirs(out, exist_ok=True)
with open(f"{out}/nav_room.pgm", "wb") as f:
    f.write(f"P5\n{w} {h}\n255\n".encode())
    f.write(bytes(grid))
with open(f"{out}/nav_room.yaml", "w") as f:
    f.write(f"image: nav_room.pgm\nmode: trinary\nresolution: {RES}\norigin: [{X0}, {Y0}, 0.0]\n"
            "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
print(f"[MAP] {w}x{h} cells @ {RES} m, origin ({X0},{Y0}) -> {out}/nav_room.yaml")
