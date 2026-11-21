#!/usr/bin/env python3
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
