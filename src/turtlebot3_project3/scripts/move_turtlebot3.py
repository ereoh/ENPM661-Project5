#!/usr/bin/env python3
"""
Closed-loop path follower for TurtleBot3.
Feed-forward wheel RPMs plus body-frame PID feedback.
• Converts cm/deg plan → m/rad
• Logs per-cycle error & command
• Uses global distance tolerance to avoid getting "stuck" when heading is off.
• Failsafe: auto-advance after 10 s at a waypoint.
"""

import os
import math
from typing import List, Tuple

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# ─────────── Constants ───────────
CONTROL_DT = 0.025          # Control loop time step (40 Hz)
R_WHEEL = 0.033             # Wheel radius in meters
L_AXLE = 0.287              # Distance between wheels in meters
RPM2RAD = 2.0 * math.pi / 60.0  # Convert RPM to radians per second

CM2M = 0.01                 # Convert centimeters to meters
DEG2RAD = math.pi / 180.0   # Convert degrees to radians

V_MAX = 0.22                # Maximum linear velocity (m/s)
W_MAX = 1.2                 # Maximum angular velocity (rad/s)

# Feedback control gains
Kp_x = 1.5                  # Proportional gain for x error
Ki_x = 0.2                  # Integral gain for x error
Kp_y = 0.3                  # Proportional gain for y error
Kp_theta = 2.0              # Proportional gain for theta error
Kd_theta = 0.3              # Derivative gain for theta error
INT_X_LIM = 0.1             # Integral windup limit for x error

# Waypoint acceptance criteria
DIST_TOL = 0.10             # Distance tolerance (meters)
ANG_TOL = 0.25              # Angular tolerance (radians)
STUCK_CYCLES = int(1.0 / CONTROL_DT)  # Timeout cycles for waypoint (1 second)

# ─────────── Utility Functions ───────────
def euler_from_quaternion(q) -> float:
    """
    Convert quaternion to yaw (Euler angle).

    Args:
        q (tuple): Quaternion (x, y, z, w).

    Returns:
        float: Yaw angle in radians.
    """
    x, y, z, w = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

# ─────────── TurtleBot3 Controller ───────────
class Turtlebot3Controller(Node):
    def __init__(self):
        """
        Initialize the TurtleBot3 controller node.
        """
        super().__init__("turtlebot3_controller")

        # Load the planned path and RPMs from the configuration file
        cfg = os.path.join(
            get_package_share_directory("turtlebot3_project3"),
            "config", "spawn_config.txt")
        self.poses, self.rpm_pairs = self._load_plan(cfg)

        # Create publishers and subscribers
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.timer = self.create_timer(CONTROL_DT, self._loop)

        # Initialize state variables
        self.i: int = 0  # Current waypoint index
        self.pose: Tuple[float, float, float] | None = None  # Current pose
        self.prev_err_th: float = 0.0  # Previous theta error
        self.err_x_int: float = 0.0  # Integrated x error
        self.cycles_at_wp: int = 0  # Cycles spent at the current waypoint

        self.get_logger().info("Closed-loop controller started")

    # ───────── Plan Loader ─────────
    @staticmethod
    def _load_plan(path: str):
        """
        Load the planned path and RPMs from a configuration file.

        Args:
            path (str): Path to the configuration file.

        Returns:
            Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]]]:
            List of poses (x, y, theta) and RPM pairs (ul, ur).
        """
        poses: List[Tuple[float, float, float]] = []
        rpms: List[Tuple[float, float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                x_cm, y_cm, th_deg, ul, ur = map(float, line.split())
                y_cm = y_cm - 150.0  # Adjust y-coordinate
                poses.append((x_cm * CM2M, y_cm * CM2M, th_deg * DEG2RAD))
                rpms.append((ul, ur))
        return poses, rpms

    # ───────── Callbacks ─────────
    def _odom_cb(self, msg: Odometry):
        """
        Odometry callback to update the robot's current pose.

        Args:
            msg (Odometry): Odometry message containing the robot's pose.
        """
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        yaw = euler_from_quaternion((o.x, o.y, o.z, o.w))
        self.pose = (p.x, p.y, yaw)

    def _loop(self):
        """
        Main control loop for closed-loop path following.
        """
        if self.pose is None:  # Wait for odometry data
            return

        if self.i >= len(self.poses):  # Stop if all waypoints are reached
            self._stop()
            return

        # Desired state (current waypoint)
        xd, yd, thd = self.poses[self.i]
        ul_rpm, ur_rpm = self.rpm_pairs[self.i]
        ul = ul_rpm * RPM2RAD
        ur = ur_rpm * RPM2RAD
        v_ff = R_WHEEL * (ul + ur) / 2.0  # Feedforward linear velocity
        w_ff = R_WHEEL * (ur - ul) / L_AXLE  # Feedforward angular velocity

        # Current pose
        x, y, th = self.pose

        # Global errors
        dx = xd - x
        dy = yd - y
        dist_err = math.hypot(dx, dy)

        # Body-frame errors for control
        err_x = math.cos(th) * dx + math.sin(th) * dy
        err_y = -math.sin(th) * dx + math.cos(th) * dy
        err_th = (thd - th + math.pi) % (2.0 * math.pi) - math.pi

        # Integrate forward error (anti-windup)
        self.err_x_int += err_x * CONTROL_DT
        self.err_x_int = max(-INT_X_LIM, min(INT_X_LIM, self.err_x_int))

        # Derivative of heading error
        d_err_th = (err_th - self.prev_err_th) / CONTROL_DT
        self.prev_err_th = err_th

        # PID control
        v_cmd = v_ff + Kp_x * err_x + Ki_x * self.err_x_int
        w_cmd = w_ff + Kp_y * err_y + Kp_theta * err_th + Kd_theta * d_err_th

        # Saturation
        v_cmd = max(-V_MAX, min(V_MAX, v_cmd))
        w_cmd = max(-W_MAX, min(W_MAX, w_cmd))

        # # Logging
        # self.get_logger().info(
        #     f"i={self.i:03d}  dist={dist_err:.3f} m  err_y={err_y:.3f} m  err_th={err_th:.3f} rad  "
        #     f"cmd=({v_cmd:.2f} m/s, {w_cmd:.2f} rad/s)")

        # Publish command
        twist = Twist()
        twist.linear.x = v_cmd
        twist.angular.z = w_cmd
        self.pub.publish(twist)

        # Waypoint acceptance check
        if dist_err < DIST_TOL and abs(err_th) < ANG_TOL:
            self._advance_wp("reached")
        else:
            self.cycles_at_wp += 1
            if self.cycles_at_wp > STUCK_CYCLES:
                self._advance_wp("timeout")

    # ───────── Helpers ─────────
    def _advance_wp(self, reason: str):
        """
        Advance to the next waypoint.

        Args:
            reason (str): Reason for advancing (e.g., "reached", "timeout").
        """
        # self.get_logger().warn(f"Advancing to waypoint {self.i+1} ({reason})")
        self.i += 1
        self.err_x_int = 0.0
        self.cycles_at_wp = 0

    def _stop(self):
        """
        Stop the robot when the path is complete.
        """
        self.get_logger().info("Path complete – stopping")
        self.timer.cancel()
        self.pub.publish(Twist())

# ───────── Main Function ─────────
def main(args=None):
    """
    Main entry point for the TurtleBot3 controller node.
    """
    rclpy.init(args=args)
    node = Turtlebot3Controller()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
