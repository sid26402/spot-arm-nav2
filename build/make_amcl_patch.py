"""AMCL step: drifting odometry (Isaac side), relay option that leaves map->odom to AMCL + publishes ground truth,
AMCL params generator, nav_up.sh --amcl, loc_err.py. Builds full_demo_perc_nav_amcl.py from full_demo_perc_nav_scan.py."""
import os, sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
N = f"{C}/nav"

def patch(path, reps, marker):
    s = open(path).read()
    if marker in s:
        print(f"[PATCH] {os.path.basename(path)} already patched"); return
    for old, new in reps:
        assert s.count(old) == 1, f"{os.path.basename(path)} pattern mismatch: {old[:70]!r}"
        s = s.replace(old, new)
    open(path, "w").write(s)
    print(f"[PATCH] {os.path.basename(path)}: {len(reps)} changes")

# 1) nav_link: scan at the TRUE pose, odometry at the DRIFTING pose, ground truth in the packet; OdomDrift model
patch(f"{N}/nav_link.py", [
    ("    def send(self, step, x, y, z, yaw, vx, vy, wz):\n        t = (step + 1) * self.dt\n        if step % self.pub_every == 0:\n            r = self.scan(x, y, yaw)\n",
     "    def send(self, step, x, y, z, yaw, vx, vy, wz, scan_pose=None, gt=None):\n"
     "        \"\"\"x, y, yaw = odometry pose (may drift). scan_pose = true pose the (physical) scan is taken from. gt = ground truth for logging.\"\"\"\n"
     "        t = (step + 1) * self.dt\n        if step % self.pub_every == 0:\n            r = self.scan(*(scan_pose if scan_pose is not None else (x, y, yaw)))\n"),
    ("                   \"ranges\": [None if not np.isfinite(v) else round(float(v), 3) for v in r]}\n        else:",
     "                   \"ranges\": [None if not np.isfinite(v) else round(float(v), 3) for v in r]}\n"
     "            if gt is not None:\n                msg[\"gx\"], msg[\"gy\"], msg[\"gyaw\"] = float(gt[0]), float(gt[1]), float(gt[2])\n        else:"),
    ("class HeadingHold:\n",
     "class OdomDrift:\n"
     "    \"\"\"Leg-odometry stand-in: integrates the measured body velocities with a scale error, a gyro yaw-rate bias and noise.\n"
     "    Starts at the true pose (odom frame = map frame at t=0), then drifts. AMCL has to correct it.\"\"\"\n"
     "    def __init__(self, dt, scale=0.03, yaw_bias=0.003, noise=0.02, seed=1):\n"
     "        self.dt, self.scale, self.yaw_bias, self.noise = dt, scale, yaw_bias, noise\n"
     "        self.rng, self.p = np.random.default_rng(seed), None\n\n"
     "    def step(self, gx, gy, gyaw, vx, vy, wz):\n"
     "        if self.p is None:\n            self.p = [gx, gy, gyaw]\n            return tuple(self.p)\n"
     "        r, k, dt = self.rng, self.noise, self.dt\n"
     "        vxm = vx * (1 + self.scale) + r.normal(0, k * abs(vx) + 1e-4)\n"
     "        vym = vy * (1 + self.scale) + r.normal(0, k * abs(vy) + 1e-4)\n"
     "        wzm = wz + self.yaw_bias + r.normal(0, k * abs(wz) + 1e-4)\n"
     "        x, y, th = self.p\n        th = wrap(th + wzm * dt)\n"
     "        x += (vxm * math.cos(th) - vym * math.sin(th)) * dt\n        y += (vxm * math.sin(th) + vym * math.cos(th)) * dt\n"
     "        self.p = [x, y, th]\n        return x, y, th\n\n\n"
     "class HeadingHold:\n"),
], "class OdomDrift")

# 2) relay: --amcl = no static map->odom (AMCL publishes it); /ground_truth (frame map) when the packet has it
patch(f"{N}/spot_ros_relay.py", [
    ("import json, math, socket\n", "import json, math, socket, sys\n"),
    ("        self.stf.sendTransform([self.tf(\"map\", \"odom\", 0.0, 0.0, 0.0, 0.0, Time()),\n                                self.tf(\"base_link\", \"laser\", 0.0, 0.0, 0.30, 0.0, Time())])\n",
     "        self.amcl = \"--amcl\" in sys.argv\n"
     "        static = [self.tf(\"base_link\", \"laser\", 0.0, 0.0, 0.30, 0.0, Time())]\n"
     "        if not self.amcl:\n            static.append(self.tf(\"map\", \"odom\", 0.0, 0.0, 0.0, 0.0, Time()))   # AMCL publishes map->odom when --amcl\n"
     "        self.stf.sendTransform(static)\n"
     "        self.gt = self.create_publisher(Odometry, \"/ground_truth\", 10)\n"),
    ("            self.odom.publish(o)\n",
     "            self.odom.publish(o)\n"
     "            if \"gx\" in m:\n"
     "                g = Odometry()\n"
     "                g.header.stamp, g.header.frame_id, g.child_frame_id = st, \"map\", \"base_link_gt\"\n"
     "                g.pose.pose.position.x, g.pose.pose.position.y = m[\"gx\"], m[\"gy\"]\n"
     "                g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(m[\"gyaw\"] / 2), math.cos(m[\"gyaw\"] / 2)\n"
     "                self.gt.publish(g)\n"),
    ("        self.get_logger().info(\"relay up: UDP 9870 <- Isaac clock/state, /cmd_vel -> UDP 9871\")\n",
     "        self.get_logger().info(f\"relay up: UDP 9870 <- Isaac clock/state, /cmd_vel -> UDP 9871 | map->odom: {'AMCL' if self.amcl else 'static identity'}\")\n"),
], "self.amcl")

# 3) AMCL params generator (initial pose from nav_world.START)
open(f"{N}/make_amcl_params.py", "w").write('''#!/usr/bin/env python3
"""Write config/nav2_params_amcl.yaml = nav2_params.yaml + an AMCL block (initial pose from nav_world.START)."""
import math, sys
sys.path.insert(0, "@SPOT_DEMO_ROOT@/nav")
import nav_world as W
N = "@SPOT_DEMO_ROOT@/nav"
base = open(f"{N}/config/nav2_params.yaml").read()
x, y, yaw = W.START[0], W.START[1], math.radians(W.START[2])
amcl = f"""
amcl:
  ros__parameters:
    use_sim_time: true
    global_frame_id: map
    odom_frame_id: odom
    base_frame_id: base_link
    scan_topic: scan
    map_topic: map
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    laser_model_type: "likelihood_field"
    laser_likelihood_max_dist: 2.0
    laser_min_range: 0.1
    laser_max_range: 10.0
    max_beams: 90
    sigma_hit: 0.2
    z_hit: 0.5
    z_short: 0.05
    z_max: 0.05
    z_rand: 0.5
    min_particles: 500
    max_particles: 2000
    pf_err: 0.05
    pf_z: 0.99
    resample_interval: 1
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    update_min_d: 0.10
    update_min_a: 0.10
    tf_broadcast: true
    transform_tolerance: 1.0
    save_pose_rate: 0.5
    first_map_only: false
    set_initial_pose: true
    initial_pose:
      x: {x}
      y: {y}
      z: 0.0
      yaw: {yaw}
"""
open(f"{N}/config/nav2_params_amcl.yaml", "w").write(base.rstrip() + "\\n" + amcl)
print(f"[AMCL] wrote {N}/config/nav2_params_amcl.yaml (initial pose {x}, {y}, {W.START[2]} deg)")
''')
print("[PATCH] make_amcl_params.py written")

# 4) loc_err.py: odometry drift and AMCL error vs ground truth
open(f"{N}/loc_err.py", "w").write('''#!/usr/bin/env python3
"""Live localisation check: odometry drift and AMCL error vs ground truth, every 2 s (sim time).
Usage: python3 loc_err.py   (ROS 2 sourced; needs the relay with ground truth + AMCL running)"""
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

def yaw_of(q):
    return 2 * math.atan2(q.z, q.w)

class LocErr(Node):
    def __init__(self):
        super().__init__("loc_err", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.gt = self.od = self.am = None
        self.create_subscription(Odometry, "/ground_truth", lambda m: setattr(self, "gt", m.pose.pose), 10)
        self.create_subscription(Odometry, "/odom", lambda m: setattr(self, "od", m.pose.pose), 10)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", lambda m: setattr(self, "am", m.pose.pose), 10)
        self.create_timer(2.0, self.report)

    def err(self, p):
        if p is None or self.gt is None:
            return "   n/a           "
        d = math.hypot(p.position.x - self.gt.position.x, p.position.y - self.gt.position.y)
        a = math.degrees(math.atan2(math.sin(yaw_of(p.orientation) - yaw_of(self.gt.orientation)),
                                    math.cos(yaw_of(p.orientation) - yaw_of(self.gt.orientation))))
        return f"{d * 100:6.1f} cm {a:+6.1f} deg"

    def report(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        g = self.gt
        pos = f"({g.position.x:+.2f},{g.position.y:+.2f})" if g else "(no ground truth yet)"
        self.get_logger().info(f"sim {t:6.1f}s true {pos} | odom drift {self.err(self.od)} | AMCL error {self.err(self.am)}")

def main():
    rclpy.init()
    rclpy.spin(LocErr())

if __name__ == "__main__":
    main()
''')
print("[PATCH] loc_err.py written")

# 5) nav_up.sh with --amcl
open(f"{N}/nav_up.sh", "w").write('''#!/bin/bash
# One terminal for the whole ROS side: relay + map_server (+ AMCL) + Nav2 + goal bridge + RViz. Ctrl+C stops all.
# Usage: nav_up.sh          -> map->odom = identity (ground-truth localisation)
#        nav_up.sh --amcl   -> AMCL localises on the lidar scan; Isaac must send drifting odometry (full_demo_perc_nav_amcl.py)
source /opt/ros/humble/setup.bash
N=@SPOT_DEMO_ROOT@/nav
L=/tmp/nav_logs
mkdir -p $L
rm -f $L/*.log
AMCL=0; [ "$1" == "--amcl" ] && AMCL=1
trap 'echo "[UP] stopping everything"; kill 0' EXIT INT TERM
if [ $AMCL == 1 ]; then
  python3 $N/make_amcl_params.py || exit 1
  P=$N/config/nav2_params_amcl.yaml
  python3 $N/spot_ros_relay.py --amcl > $L/relay.log 2>&1 &
else
  P=$N/config/nav2_params.yaml
  python3 $N/spot_ros_relay.py > $L/relay.log 2>&1 &
fi
sleep 3
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=$N/maps/nav_room.yaml -p use_sim_time:=true > $L/map.log 2>&1 &
if [ $AMCL == 1 ]; then
  ros2 run nav2_amcl amcl --ros-args --params-file $P > $L/amcl.log 2>&1 &
  ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_localization \\
    -p node_names:="[map_server, amcl]" -p autostart:=true -p use_sim_time:=true > $L/map_lm.log 2>&1 &
else
  ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_map \\
    -p node_names:="[map_server]" -p autostart:=true -p use_sim_time:=true > $L/map_lm.log 2>&1 &
fi
sleep 3
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true use_composition:=False autostart:=true \\
  params_file:=$P > $L/nav2.log 2>&1 &
python3 $N/goal_bridge.py > $L/bridge.log 2>&1 &
rviz2 -d /opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz --ros-args -p use_sim_time:=true > $L/rviz.log 2>&1 &
echo "[UP] started relay, map_server$([ $AMCL == 1 ] && echo ', AMCL'), Nav2, goal bridge, RViz - params $P - logs in $L  (Ctrl+C stops all)"
tail -F $L/relay.log $L/map_lm.log $L/amcl.log $L/nav2.log $L/bridge.log 2>/dev/null | grep --line-buffered -iE "relay up|bridge up|goal [0-9]+|sim t=|nodes are active|initial pose|error|fail|abort|succeed|exception"
''')
os.chmod(f"{N}/nav_up.sh", 0o755)
print("[PATCH] nav_up.sh rewritten (--amcl option)")

# 6) demo: drifting odometry
src, dst = f"{C}/full_demo_perc_nav_scan.py", f"{C}/full_demo_perc_nav_amcl.py"
s = open(src).read()
R = [
 ('parser.add_argument("--scan_test", action="store_true", help="SCAN: compare physx scan vs box scan at the start pose, then quit")\n',
  'parser.add_argument("--scan_test", action="store_true", help="SCAN: compare physx scan vs box scan at the start pose, then quit")\n'
  'parser.add_argument("--no_drift", action="store_true", help="ODOM: send exact odometry (no drift)")\n'
  'parser.add_argument("--odom_scale", type=float, default=0.03, help="ODOM: speed scale error (0.03 = odometry reads 3%% long)")\n'
  'parser.add_argument("--odom_yaw_bias", type=float, default=0.003, help="ODOM: gyro yaw-rate bias (rad/s); 0.003 = 0.17 deg/s")\n'
  'parser.add_argument("--odom_noise", type=float, default=0.02, help="ODOM: velocity noise, fraction of speed")\n'),
 ('from nav_link import StateLink, HeadingHold, GoalClient, Scan2D\n', 'from nav_link import StateLink, HeadingHold, GoalClient, Scan2D, OdomDrift\n'),
 ('link = StateLink(W.OBSTACLES, dt, scan_fn=_scan)\n',
  'link = StateLink(W.OBSTACLES, dt, scan_fn=_scan)\n'
  'odom = OdomDrift(dt, scale=args.odom_scale, yaw_bias=args.odom_yaw_bias, noise=args.odom_noise)\n'
  'print(f"[ODOM] {\'exact (no drift)\' if args.no_drift else f\'drifting: scale {args.odom_scale:+.0%}, yaw bias {math.degrees(args.odom_yaw_bias):.2f} deg/s, noise {args.odom_noise:.0%}\'}"\n'
  '      f" - Nav2 localisation needs nav_up.sh --amcl", flush=True)\n'),
 ('            link.send(step, float(_pw[0] - origin[0]), float(_pw[1] - origin[1]), float(_pw[2]), float(_yw[0]),\n'
  '                      float(_lv[0]), float(_lv[1]), float(_av[2]))\n',
  '            _gx, _gy, _gyaw = float(_pw[0] - origin[0]), float(_pw[1] - origin[1]), float(_yw[0])\n'
  '            _vx, _vy, _wz = float(_lv[0]), float(_lv[1]), float(_av[2])\n'
  '            _ox, _oy, _oyaw = (_gx, _gy, _gyaw) if args.no_drift else odom.step(_gx, _gy, _gyaw, _vx, _vy, _wz)\n'
  '            link.send(step, _ox, _oy, float(_pw[2]), _oyaw, _vx, _vy, _wz, scan_pose=(_gx, _gy, _gyaw), gt=(_gx, _gy, _gyaw))\n'
  '            if step % int(10.0 / dt) == 0:\n'
  '                print(f"[ODOM] t={step * dt:.0f}s true ({_gx:+.2f},{_gy:+.2f},{math.degrees(_gyaw):+.0f}) odom ({_ox:+.2f},{_oy:+.2f},{math.degrees(_oyaw):+.0f}) "\n'
  '                      f"drift {math.hypot(_ox - _gx, _oy - _gy) * 100:.0f} cm {math.degrees(math.atan2(math.sin(_oyaw - _gyaw), math.cos(_oyaw - _gyaw))):+.1f} deg", flush=True)\n'),
]
for old, new in R:
    n = s.count(old)
    assert n == 1, f"[MERGE] FAILED: pattern found {n} times (need 1): {old[:80]!r}"
    s = s.replace(old, new)
open(dst, "w").write(s)
print(f"[MERGE] OK: {len(R)} changes -> {dst}")
