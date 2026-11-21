"""Simulated 2D lidar for Isaac Lab: PhysX ray casts against the real scene colliders (like Isaac Sim's PhysX lidar).
Sees anything with a collider - walls, bench, bin, and objects that are NOT in the Nav2 map.
Spot's own colliders (and any other excluded prim paths) are ignored. Import only inside the Isaac app."""
import math
import numpy as np
import carb
from omni.physx import get_physx_scene_query_interface


class PhysxScan:
    def __init__(self, origin_xy, exclude, n_rays=360, height=0.12, r_min=0.10, r_max=10.0):
        """origin_xy: env origin in world (scan is called with env-relative x, y like Scan2D).
        exclude: tuple of prim-path prefixes to ignore (Spot itself). height: world z of the scan plane (m)."""
        self.o = (float(origin_xy[0]), float(origin_xy[1]))
        self.excl = tuple(exclude)
        self.h, self.r_min, self.r_max = float(height), float(r_min), float(r_max)
        self.angles = np.linspace(-math.pi, math.pi, n_rays, endpoint=False)   # same convention as Scan2D / the relay
        self.sq = get_physx_scene_query_interface()
        self._best = math.inf
        self.hit_paths = {}                       # prim path -> hit count (for the --scan_test report)
        self.raw_hits, self.robot_hits = 0, 0     # every hit PhysX reports / hits on Spot itself (diagnostics)

    def _cb(self, hit):
        self.raw_hits += 1
        p = str(getattr(hit, "collision", ""))
        if p.startswith(self.excl):
            self.robot_hits += 1
            return True                           # Spot itself: ignore, keep looking
        d = float(hit.distance)
        if self.r_min <= d < self._best:
            self._best = d
            self._path = p
        return True

    def __call__(self, x, y, yaw):
        ox, oy = x + self.o[0], y + self.o[1]
        orig = carb.Float3(ox, oy, self.h)
        out = np.full(len(self.angles), np.inf)
        for i, a in enumerate(self.angles):
            self._best, self._path = math.inf, None
            self.sq.raycast_all(orig, carb.Float3(math.cos(a + yaw), math.sin(a + yaw), 0.0), self.r_max, self._cb)
            out[i] = self._best
            if self._path is not None:
                self.hit_paths[self._path] = self.hit_paths.get(self._path, 0) + 1
        return out
