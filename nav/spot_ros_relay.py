#!/usr/bin/env python3
"""ROS2 Humble relay for the Isaac Spot sim: UDP 9870 (clock every step, state+scan at 25 Hz) -> /clock /odom /tf /scan ;
/cmd_vel -> UDP 9871."""
import json, math, socket, sys
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster


def stamp(t):
    s = int(t)
    return Time(sec=s, nanosec=int((t - s) * 1e9))


class Relay(Node):
    def __init__(self):
        super().__init__("spot_isaac_relay")
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("127.0.0.1", 9870))
        self.rx.setblocking(False)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.isaac = ("127.0.0.1", 9871)
        self.clock = self.create_publisher(Clock, "/clock", 10)
        self.odom = self.create_publisher(Odometry, "/odom", 10)
        self.scan = self.create_publisher(LaserScan, "/scan", 10)
        self.tfb = TransformBroadcaster(self)
        self.stf = StaticTransformBroadcaster(self)
        self.amcl = "--amcl" in sys.argv
        static = [self.tf("base_link", "laser", 0.0, 0.0, 0.30, 0.0, Time())]
        if not self.amcl:
            static.append(self.tf("map", "odom", 0.0, 0.0, 0.0, 0.0, Time()))   # AMCL publishes map->odom when --amcl
        self.stf.sendTransform(static)
        self.gt = self.create_publisher(Odometry, "/ground_truth", 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.create_timer(0.002, self.poll)
        self.n = 0
        self.get_logger().info(f"relay up: UDP 9870 <- Isaac clock/state, /cmd_vel -> UDP 9871 | map->odom: {'AMCL' if self.amcl else 'static identity'}")

    def tf(self, parent, child, x, y, z, yaw, st):
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = st, parent, child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
        t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return t

    def on_cmd(self, m):
        self.tx.sendto(json.dumps({"vx": m.linear.x, "vy": m.linear.y, "wz": m.angular.z}).encode(), self.isaac)

    def poll(self):
        while True:
            try:
                d = self.rx.recv(65535)
            except BlockingIOError:
                return
            m = json.loads(d)
            st = stamp(m["t"])
            self.clock.publish(Clock(clock=st))
            if "x" not in m:          # clock-only packet (every sim step)
                continue
            o = Odometry()
            o.header.stamp, o.header.frame_id, o.child_frame_id = st, "odom", "base_link"
            o.pose.pose.position.x, o.pose.pose.position.y = m["x"], m["y"]
            o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(m["yaw"] / 2), math.cos(m["yaw"] / 2)
            o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.angular.z = m["vx"], m["vy"], m["wz"]
            self.odom.publish(o)
            if "gx" in m:
                g = Odometry()
                g.header.stamp, g.header.frame_id, g.child_frame_id = st, "map", "base_link_gt"
                g.pose.pose.position.x, g.pose.pose.position.y = m["gx"], m["gy"]
                g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(m["gyaw"] / 2), math.cos(m["gyaw"] / 2)
                self.gt.publish(g)
            self.tfb.sendTransform(self.tf("odom", "base_link", m["x"], m["y"], 0.0, m["yaw"], st))
            s = LaserScan()
            s.header.stamp, s.header.frame_id = st, "laser"
            n = len(m["ranges"])
            s.angle_min, s.angle_increment = -math.pi, 2 * math.pi / n
            s.angle_max = s.angle_min + s.angle_increment * (n - 1)
            s.range_min, s.range_max = 0.1, 10.0
            s.ranges = [float("inf") if v is None else float(v) for v in m["ranges"]]
            self.scan.publish(s)
            self.n += 1
            if self.n % 125 == 0:
                self.get_logger().info(f"sim t={m['t']:.1f}s  pos=({m['x']:+.2f},{m['y']:+.2f})  yaw={math.degrees(m['yaw']):+.0f}")


def main():
    rclpy.init()
    rclpy.spin(Relay())


if __name__ == "__main__":
    main()
