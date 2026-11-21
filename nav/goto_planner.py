"""General go_to planning for Spot: stand pose + pre-pose for any target and approach direction, checked against the map.
Pure python + numpy. Usable from Isaac (py3.12), system python, and the perception/manipulation side.
  stand pose : where Spot must finally stand (target `standoff` m straight ahead), reached by the scripted fine approach
  pre-pose   : Nav2 goal, `pre_dist` m behind the stand pose on the same heading; must leave room to turn in place
plan() tries each candidate heading, rejects unsafe ones with a reason, and returns the valid one closest to the robot."""
import math
import numpy as np

HALF_L, HALF_W = 0.55, 0.25          # Spot footprint half-length / half-width (arm stowed), same as Nav2


class GotoPlanner:
    def __init__(self, obstacles, pre_dist=0.5, turn_r=0.65, max_extra=0.5):
        """obstacles: nav_world-style boxes (name, cx, cy, sx, sy, h, spawn). turn_r: centre-to-obstacle distance needed
        to turn in place (corner radius 0.60 + margin). max_extra: how much further back the pre-pose may move to find room."""
        self.boxes = [(n, cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2) for n, cx, cy, sx, sy, *_ in obstacles]
        self.pre_dist, self.turn_r, self.max_extra = pre_dist, turn_r, max_extra

    def dist(self, x, y):
        """Distance from a point to the nearest obstacle (0 inside one) and its name."""
        best, who = math.inf, None
        for n, x0, x1, y0, y1 in self.boxes:
            d = math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))
            if d < best:
                best, who = d, n
        return best, who

    def fp_clear(self, x, y, yaw):
        """Clearance of Spot's footprint outline at (x, y, yaw); negative = overlapping an obstacle."""
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [(HALF_L, HALF_W), (HALF_L, -HALF_W), (-HALF_L, -HALF_W), (-HALF_L, HALF_W)]
        best, who = math.inf, None
        for k in range(4):
            (ax, ay), (bx, by) = corners[k], corners[(k + 1) % 4]
            for t in np.linspace(0.0, 1.0, 12):
                px, py = ax + (bx - ax) * t, ay + (by - ay) * t
                d, n = self.dist(x + c * px - s * py, y + s * px + c * py)
                if d < best:
                    best, who = d, n
        if best == 0.0:
            best = -0.01
        return best, who

    def plan(self, target_xy, standoff, headings, robot_xy=None, stand_clear=-0.03, label=""):
        """Returns dict(ok, stand=(x, y, yaw), pre=(x, y, yaw), reason, tried=[...]).
        headings: candidate approach directions (rad) = direction Spot faces at the stand pose."""
        tx, ty = target_xy
        tried, valid = [], []
        for h in headings:
            ux, uy = math.cos(h), math.sin(h)
            sx, sy = tx - standoff * ux, ty - standoff * uy
            c_stand, who = self.fp_clear(sx, sy, h)
            tag = f"heading {math.degrees(h):+4.0f}"
            if c_stand < stand_clear:
                tried.append(f"{tag}: stand pose overlaps {who} by {-c_stand * 100:.0f} cm"); continue
            ok_pre = None
            for extra in np.arange(0.0, self.max_extra + 1e-6, 0.1):
                px, py = sx - (self.pre_dist + extra) * ux, sy - (self.pre_dist + extra) * uy
                d_pre, who_pre = self.dist(px, py)
                corridor = min(self.fp_clear(sx + (px - sx) * t, sy + (py - sy) * t, h)[0] for t in np.linspace(0, 1, 8))
                if d_pre >= self.turn_r and corridor >= stand_clear:
                    ok_pre = (px, py, extra); break
            if ok_pre is None:
                tried.append(f"{tag}: no room to turn behind the stand pose (nearest {who_pre} {d_pre:.2f} m < {self.turn_r})"); continue
            px, py, extra = ok_pre
            far = math.hypot(px - robot_xy[0], py - robot_xy[1]) if robot_xy is not None else 0.0
            tried.append(f"{tag}: OK (stand clearance {c_stand * 100:+.0f} cm, pre-pose {self.pre_dist + extra:.1f} m back, {far:.1f} m from robot)")
            valid.append((far, (sx, sy, h), (px, py, h)))
        if not valid:
            return {"ok": False, "stand": None, "pre": None, "tried": tried,
                    "reason": f"{label}: no reachable approach to ({tx:+.2f},{ty:+.2f})"}
        far, stand, pre = min(valid, key=lambda v: v[0])
        return {"ok": True, "stand": stand, "pre": pre, "tried": tried,
                "reason": f"{label}: approach heading {math.degrees(stand[2]):+.0f} deg"}


def pick_headings(bench):
    """Approach from either long side of the bench (the short sides leave Spot's nose over the bench)."""
    return [math.pi / 2, -math.pi / 2] if bench[3] > bench[4] else [0.0, math.pi]


def place_headings():
    return [k * math.pi / 4 for k in range(8)]
