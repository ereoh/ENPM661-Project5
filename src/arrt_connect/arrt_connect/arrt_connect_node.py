#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

# Import planning and visualization
from search import arrt_connect, visualize_path_and_trees

CONTROL_DT = 0.1  # seconds
WAYPOINT_TOL = 5.0  # cm
LINEAR_SPEED = 0.1  # m/s
ANGULAR_SPEED = 0.5  # rad/s

def smooth_path(path, weight_data=0.5, weight_smooth=0.25, tolerance=0.01):
    """
    Smooths the path using gradient descent.
    :param path: List of waypoints [(x1, y1), (x2, y2), ...]
    :param weight_data: Weight for the original data points
    :param weight_smooth: Weight for the smoothing term
    :param tolerance: Convergence tolerance
    :return: Smoothed path
    """
    new_path = [list(point) for point in path]
    change = tolerance
    while change >= tolerance:
        change = 0.0
        for i in range(1, len(path) - 1):  # Skip the first and last points
            for j in range(len(path[i])):
                aux = new_path[i][j]
                new_path[i][j] += weight_data * (path[i][j] - new_path[i][j])
                new_path[i][j] += weight_smooth * (new_path[i - 1][j] + new_path[i + 1][j] - 2.0 * new_path[i][j])
                change += abs(aux - new_path[i][j])
    return [tuple(point) for point in new_path]

class TurtlebotWaypointNode(Node):
    def __init__(self):
        super().__init__('turtlebot_waypoint_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pose = None
        self.timer = self.create_timer(CONTROL_DT, self.control_loop)

        # ───── Get path from ARRT-Connect ─────
        start = (50, 250)   # cm
        goal = (500, 250)  # cm
        max_iter = 25_000
        pgoal = 0.2  # Probability of sampling qgoal
        poutside = 0.95  # Range for random sampling

        path, tree_a, tree_b = arrt_connect(start, goal, max_iter, pgoal, poutside)

        # path.reverse()

        # print(f"Path found: {path}")

        # Smooth the path
        path = smooth_path(path)
        # print(f"Smoothed path: {path}")

        # ───── Visualize search and path ─────
        visualize_path_and_trees(tree_a, tree_b, path)

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
        # if cmd != self.previous_cmd:
        #     print(f"Sending command: {cmd}")
        #     print(f"Current pose: {self.pose}")
        #     print(f"Target waypoint: {self.path[self.index]}")
        #     print(f"Angle difference: {angle_diff}")
        #     print(f"Distance to waypoint: {dist}")
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
