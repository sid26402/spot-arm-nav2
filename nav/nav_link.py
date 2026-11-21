"""Isaac-side link to the ROS2 nav stack. Pure python + numpy (no ROS), importable from Isaac (py3.12) and system python.
StateLink : clock every step + state/scan at `rate` -> relay (UDP 9870); cmd_vel <- relay (UDP 9871), sim-time watchdog.
HeadingHold: holds yaw when the command is zero (walking policy drifts ~2 deg/s), adds the policy's wz bias.
GoalClient: Nav2 goals -> goal_bridge (UDP 9872); status <- bridge (UDP 9873)."""
import json, math, socket, time
import numpy as np

RELAY, CMD_PORT, BRIDGE, REPLY_PORT = ("127.0.0.1", 9870), 9871, ("127.0.0.1", 9872), 9873
TERMINAL = ("SUCCEEDED", "ABORTED", "CANCELED", "REJECTED")


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Scan2D:
    def __init__(self, obstacles, n_rays=360, r_max=10.0):
        self.angles = np.linspace(-math.pi, math.pi, n_rays, endpoint=False)
        self.r_max = r_max
        self.boxes = np.array([[cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2] for _, cx, cy, sx, sy, _, _ in obstacles])

    def __call__(self, x, y, yaw):
        d = np.stack([np.cos(self.angles + yaw), np.sin(self.angles + yaw)], 1)
        inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
        b = self.boxes
        tx1, tx2 = (b[None, :, 0] - x) * inv[:, :1], (b[None, :, 1] - x) * inv[:, :1]
        ty1, ty2 = (b[None, :, 2] - y) * inv[:, 1:], (b[None, :, 3] - y) * inv[:, 1:]
        tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
        tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
        r = np.where((tmax >= tmin) & (tmin > 0.0), tmin, np.inf).min(1)
        return np.where(r > self.r_max, np.inf, r)


class StateLink:
    def __init__(self, obstacles, dt, rate=25.0, watchdog_s=0.5, lim=(1.0, 0.5, 1.0), scan_fn=None):
        self.dt, self.lim = dt, lim
        self.scan = Scan2D(obstacles) if scan_fn is None else scan_fn   # scan_fn(x, y, yaw) -> ranges, e.g. PhysxScan
        self.last = None
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("127.0.0.1", CMD_PORT))
        self.rx.setblocking(False)
        self.pub_every = max(1, int(round(1.0 / (rate * dt))))
        self.wd_steps = int(round(watchdog_s / dt))
        self.cmd, self.last_cmd_step = [0.0, 0.0, 0.0], -10**9

    def recv_cmd(self, step):
        while True:
            try:
                m = json.loads(self.rx.recv(4096))
                L = self.lim
                self.cmd = [max(-L[0], min(L[0], float(m["vx"]))), max(-L[1], min(L[1], float(m["vy"]))), max(-L[2], min(L[2], float(m["wz"])))]
                self.last_cmd_step = step
            except BlockingIOError:
                break
            except Exception:
                break
        if step - self.last_cmd_step > self.wd_steps:
            self.cmd = [0.0, 0.0, 0.0]
        return list(self.cmd)

    def send(self, step, x, y, z, yaw, vx, vy, wz, scan_pose=None, gt=None):
        """x, y, yaw = odometry pose (may drift). scan_pose = true pose the (physical) scan is taken from. gt = ground truth for logging."""
        t = (step + 1) * self.dt
        if step % self.pub_every == 0:
            r = self.scan(*(scan_pose if scan_pose is not None else (x, y, yaw)))
            self.last = r
            msg = {"t": t, "x": x, "y": y, "z": z, "yaw": yaw, "vx": vx, "vy": vy, "wz": wz,
                   "ranges": [None if not np.isfinite(v) else round(float(v), 3) for v in r]}
            if gt is not None:
                msg["gx"], msg["gy"], msg["gyaw"] = float(gt[0]), float(gt[1]), float(gt[2])
        else:
            msg = {"t": t}
        self.tx.sendto(json.dumps(msg).encode(), RELAY)


class OdomDrift:
    """Leg-odometry stand-in: integrates the measured body velocities with a scale error, a gyro yaw-rate bias and noise.
    Starts at the true pose (odom frame = map frame at t=0), then drifts. AMCL has to correct it."""
    def __init__(self, dt, scale=0.03, yaw_bias=0.003, noise=0.02, seed=1):
        self.dt, self.scale, self.yaw_bias, self.noise = dt, scale, yaw_bias, noise
        self.rng, self.p = np.random.default_rng(seed), None

    def step(self, gx, gy, gyaw, vx, vy, wz):
        if self.p is None:
            self.p = [gx, gy, gyaw]
            return tuple(self.p)
        r, k, dt = self.rng, self.noise, self.dt
        vxm = vx * (1 + self.scale) + r.normal(0, k * abs(vx) + 1e-4)
        vym = vy * (1 + self.scale) + r.normal(0, k * abs(vy) + 1e-4)
        wzm = wz + self.yaw_bias + r.normal(0, k * abs(wz) + 1e-4)
        x, y, th = self.p
        th = wrap(th + wzm * dt)
        x += (vxm * math.cos(th) - vym * math.sin(th)) * dt
        y += (vxm * math.sin(th) + vym * math.cos(th)) * dt
        self.p = [x, y, th]
        return x, y, th


class HeadingHold:
    def __init__(self, k=3.0, wz_bias=0.10, wz_max=0.3):
        self.k, self.bias, self.wz_max, self.yaw_hold = k, wz_bias, wz_max, None

    def __call__(self, cmd, yaw):
        """cmd = Nav2 [vx, vy, wz] -> command for the walking policy."""
        if max(abs(c) for c in cmd) < 1e-3:
            if self.yaw_hold is None:
                self.yaw_hold = yaw
            e = wrap(yaw - self.yaw_hold)
            return [0.0, 0.0, max(-self.wz_max, min(self.wz_max, -self.k * e)) + self.bias]
        self.yaw_hold = None
        return [cmd[0], cmd[1], cmd[2] + self.bias]


class GoalClient:
    def __init__(self, resend_s=1.0):
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("127.0.0.1", REPLY_PORT))
        self.rx.setblocking(False)
        self.resend_s, self.id, self.req = resend_s, int(time.time()) % 100000, None
        self.status, self.dist, self.last_send = "IDLE", -1.0, -1e9

    def go(self, x, y, yaw):
        self.id += 1
        self.req = {"id": self.id, "x": float(x), "y": float(y), "yaw": float(yaw)}
        self.status, self.dist, self.last_send = "PENDING", -1.0, -1e9
        return self.id

    def cancel(self):
        if self.req is not None and self.status not in TERMINAL:
            self.tx.sendto(json.dumps({"id": self.id, "cancel": True}).encode(), BRIDGE)

    def update(self):
        while True:
            try:
                m = json.loads(self.rx.recv(4096))
            except BlockingIOError:
                break
            except Exception:
                break
            if m.get("id") == self.id:
                self.status, self.dist = m["status"], float(m.get("dist", -1.0))
        now = time.monotonic()
        if self.req is not None and self.status in ("PENDING", "SENT", "NO_SERVER") and now - self.last_send > self.resend_s:
            self.tx.sendto(json.dumps(self.req).encode(), BRIDGE)
            self.last_send = now
        return self.status

    @property
    def done(self):
        return self.status in TERMINAL
