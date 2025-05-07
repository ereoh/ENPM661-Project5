#!/usr/bin/env python3

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

# Import planning and visualization
from search import arrt_anytime_connect, visualize_search, visualize_path

CONTROL_DT = 0.5  # seconds
WAYPOINT_TOL = 5.0  # cm
LINEAR_SPEED = 0.1  # m/s
ANGULAR_SPEED = 0.5  # rad/s

class TurtlebotWaypointNode(Node):
    def __init__(self):
        super().__init__('turtlebot_waypoint_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pose = None
        self.timer = self.create_timer(CONTROL_DT, self.control_loop)

        # ───── Get path from ARRT-Connect ─────
        start = (50, 250)   # cm
        goal = (250, 200)  # cm
        step = 5
        max_iter = 5000
        buffer = 10

        path, tree_a, tree_b = arrt_anytime_connect(start, goal, step, max_iter, buffer)

        path.reverse()

        print(f"Path found: {path}")

        # ───── Visualize search and path ─────
        visualize_search(path, start, goal, tree_a, tree_b, buffer)
        visualize_path(path, start, goal, buffer)

        self.path = path
        self.index = 0
        self.previous_cmd = Twist()

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.pose = (pos.x * 100.0, (pos.y + 1.5) * 100.0, yaw)  # convert m to cm

    def control_loop(self):
        if self.pose is None or self.index >= len(self.path):
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        x, y, theta = self.pose
        tx, ty = self.path[self.index]
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)

        if dist < WAYPOINT_TOL:
            self.index += 1
            return

        target_theta = math.atan2(dy, dx)
        angle_diff = self.normalize_angle(target_theta - theta)

        cmd = Twist()
        if cmd != self.previous_cmd:
            print(f"Sending command: {cmd}")
            print(f"Current pose: {self.pose}")
            print(f"Target waypoint: {self.path[self.index]}")
            print(f"Angle difference: {angle_diff}")
            print(f"Distance to waypoint: {dist}")
        self.previous_cmd = cmd
        if abs(angle_diff) > 0.2:
            cmd.angular.z = ANGULAR_SPEED if angle_diff > 0 else -ANGULAR_SPEED
        else:
            cmd.linear.x = LINEAR_SPEED

        self.cmd_pub.publish(cmd)

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotWaypointNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
