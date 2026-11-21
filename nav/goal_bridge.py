#!/usr/bin/env python3
"""Nav2 goal bridge (system ROS2 Humble): Isaac goal requests on UDP 9872 -> NavigateToPose action;
status + distance_remaining back to Isaac on UDP 9873.
Request: {"id": n, "x": m, "y": m, "yaw": rad}   or   {"id": n, "cancel": true}
Reply:   {"id": n, "status": "SENT|ACTIVE|SUCCEEDED|ABORTED|CANCELED|CANCELING|REJECTED|NO_SERVER", "dist": m}"""
import json, math, socket
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

FINAL = {GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED", GoalStatus.STATUS_ABORTED: "ABORTED", GoalStatus.STATUS_CANCELED: "CANCELED"}


class Bridge(Node):
    def __init__(self):
        super().__init__("nav_goal_bridge")
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("127.0.0.1", 9872))
        self.rx.setblocking(False)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.isaac = ("127.0.0.1", 9873)
        self.ac = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.cur_id, self.status, self.dist, self.goal_handle = None, "IDLE", -1.0, None
        self.create_timer(0.01, self.poll)
        self.create_timer(0.5, self.report)
        self.get_logger().info("goal bridge up: UDP 9872 -> navigate_to_pose, status -> UDP 9873")

    def reply(self, gid=None, status=None):
        msg = {"id": self.cur_id if gid is None else gid, "status": self.status if status is None else status, "dist": self.dist}
        self.tx.sendto(json.dumps(msg).encode(), self.isaac)

    def poll(self):
        while True:
            try:
                m = json.loads(self.rx.recv(4096))
            except BlockingIOError:
                return
            except Exception:
                return
            if m.get("cancel"):
                if m["id"] == self.cur_id and self.goal_handle is not None and self.status in ("SENT", "ACTIVE"):
                    self.goal_handle.cancel_goal_async()
                    self.status = "CANCELING"
                    self.get_logger().info(f"goal {self.cur_id}: cancel requested")
                self.reply()
                continue
            if m["id"] != self.cur_id:
                if not self.ac.server_is_ready():
                    self.reply(m["id"], "NO_SERVER")
                    continue
                self.send_goal(m)
            self.reply()

    def send_goal(self, m):
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = "map"
        g.pose.pose.position.x, g.pose.pose.position.y = float(m["x"]), float(m["y"])
        g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(m["yaw"] / 2), math.cos(m["yaw"] / 2)
        self.cur_id, self.status, self.dist, self.goal_handle = m["id"], "SENT", -1.0, None
        self.get_logger().info(f"goal {m['id']}: ({m['x']:+.2f},{m['y']:+.2f}) yaw {math.degrees(m['yaw']):+.0f} deg -> Nav2")
        fut = self.ac.send_goal_async(g, feedback_callback=self.on_fb)
        fut.add_done_callback(lambda f, gid=m["id"]: self.on_accept(f, gid))

    def on_accept(self, f, gid):
        if gid != self.cur_id:
            return
        h = f.result()
        if not h.accepted:
            self.status = "REJECTED"
            self.get_logger().warn(f"goal {gid}: REJECTED")
            self.reply()
            return
        self.goal_handle, self.status = h, "ACTIVE"
        h.get_result_async().add_done_callback(lambda r, gid=gid: self.on_result(r, gid))

    def on_fb(self, fb):
        self.dist = float(fb.feedback.distance_remaining)

    def on_result(self, r, gid):
        if gid != self.cur_id:
            return
        st = r.result().status
        self.status = FINAL.get(st, f"STATUS_{st}")
        self.get_logger().info(f"goal {gid}: {self.status}")
        self.reply()

    def report(self):
        if self.cur_id is not None:
            self.reply()


def main():
    rclpy.init()
    rclpy.spin(Bridge())


if __name__ == "__main__":
    main()
