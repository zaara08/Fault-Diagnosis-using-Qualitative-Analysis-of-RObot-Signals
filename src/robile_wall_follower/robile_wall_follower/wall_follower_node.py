#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math

# ── TUNING PARAMETERS ──────────────────────────────────────────
TARGET_DISTANCE  = 0.5   # metres from wall (right side)
KP               = 1.2   # proportional gain (how strongly to correct)
FORWARD_SPEED    = 0.1   # m/s forward speed (slow and safe)
MAX_ANGULAR      = 0.5   # maximum turning speed (rad/s)
WALL_FOUND_DIST  = 2.0   # if right side < this, wall is detected
DANGER_FRONT     = 0.25  # emergency stop if front closer than this
PILLAR_THRESHOLD = 0.35  # if wall suddenly < this, pillar detected
# ───────────────────────────────────────────────────────────────


class WallFollowerNode(Node):
    """
    Wall follower using a P-controller.
    The robot keeps the wall on its RIGHT side at TARGET_DISTANCE.
    Works for both plain walls and walls with pillars.
    """

    def __init__(self):
        super().__init__('wall_follower_node')

        # --- Publisher: sends movement commands to robot ---
        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)

        # --- Publisher: sends action label to data logger ---
        self.label_pub = self.create_publisher(
            String, '/action_label', 10)

        # --- Subscriber: receives LIDAR data ---
        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # --- Timer: runs control loop every 0.1 seconds ---
        self.create_timer(0.1, self.control_loop)

        # --- Internal state variables ---
        self.lidar_right       = None   # distance to right wall
        self.lidar_front_right = None   # distance front-right
        self.lidar_front       = None   # distance straight ahead
        self.action_label      = 'idle' # current action label

        self.get_logger().info('Wall Follower Node started!')
        self.get_logger().info(
            f'Target distance: {TARGET_DISTANCE}m | '
            f'Speed: {FORWARD_SPEED}m/s')


    def scan_callback(self, msg):
        """
        Called every time a new LIDAR scan arrives.
        Extracts 3 key distances from the full scan.

        The LIDAR scans from -90deg (right) to +90deg (left).
        We extract:
          - right      (index for -90 deg)
          - front_right (index for -45 deg)
          - front      (index for   0 deg)
        """
        ranges = msg.ranges
        total  = len(ranges)

        # Helper to safely get a LIDAR value
        # (some beams return 'inf' if nothing detected)
        def safe_get(index):
            val = ranges[index]
            if math.isinf(val) or math.isnan(val):
                return msg.range_max
            return val

        # Extract the three key angles
        # index 0   = rightmost (-90 deg)
        # index mid = front (0 deg)
        # index end = leftmost (+90 deg)
        right_idx       = 0
        front_right_idx = total // 4       # -45 degrees
        front_idx       = total // 2       #   0 degrees

        self.lidar_right       = safe_get(right_idx)
        self.lidar_front_right = safe_get(front_right_idx)
        self.lidar_front       = safe_get(front_idx)


    def get_action_label(self, right, front):
        """
        Decides what the robot is currently doing
        based on LIDAR readings.
        Returns a string label for the CSV.
        """
        # Emergency: something very close in front
        if front < DANGER_FRONT:
            return 'wall_following_fail'

        # No wall detected on right side yet
        if right > WALL_FOUND_DIST:
            return 'searching_wall'

        # Approaching: robot is far from wall
        if right > TARGET_DISTANCE + 0.15:
            return 'approaching_wall'

        # Pillar detected: wall suddenly very close
        if right < PILLAR_THRESHOLD:
            return 'wall_avoiding_pillar'

        # Too close (but not pillar)
        if right < TARGET_DISTANCE - 0.15:
            return 'wall_following_fail'

        # Perfect: robot is at target distance
        return 'wall_following'


    def control_loop(self):
        """
        Runs every 0.1 seconds.
        Reads LIDAR values, calculates correction,
        publishes movement command and action label.
        """
        # Wait until first LIDAR reading arrives
        if self.lidar_right is None:
            return

        right = self.lidar_right
        front = self.lidar_front
        front_right = self.lidar_front_right

        # ── SAFETY CHECK ───────────────────────────────────
        # If something is dangerously close in front → STOP
        if front < DANGER_FRONT:
            self.stop_robot()
            self.action_label = 'wall_following_fail'
            self.publish_label()
            self.get_logger().warn(
                f'EMERGENCY STOP! Front obstacle: {front:.2f}m')
            return

        # ── SEARCHING ──────────────────────────────────────
        # No wall on right side → rotate slowly to find it
        if right > WALL_FOUND_DIST:
            cmd = Twist()
            cmd.linear.x  = 0.0
            cmd.angular.z = -0.3   # rotate right to find wall
            self.cmd_pub.publish(cmd)
            self.action_label = 'searching_wall'
            self.publish_label()
            return

        # ── P-CONTROLLER ───────────────────────────────────
        # error = how far we are from target distance
        # positive error = too far from wall → turn right
        # negative error = too close to wall → turn left
        error = right - TARGET_DISTANCE

        # Special case: pillar approaching (front_right close)
        # Steer away more aggressively
        if front_right < PILLAR_THRESHOLD:
            error = -0.3   # force left turn to avoid pillar

        # Calculate angular correction
        angular = KP * error
        # Clamp to maximum turning speed
        angular = max(-MAX_ANGULAR, min(MAX_ANGULAR, angular))

        # Build and send movement command
        cmd = Twist()
        cmd.linear.x  = FORWARD_SPEED
        cmd.angular.z = -angular   # negative = turn right

        self.cmd_pub.publish(cmd)

        # ── ACTION LABEL ───────────────────────────────────
        self.action_label = self.get_action_label(right, front)
        self.publish_label()

        # ── LOGGING ────────────────────────────────────────
        self.get_logger().info(
            f'R:{right:.2f}m FR:{front_right:.2f}m '
            f'F:{front:.2f}m | err:{error:.2f} '
            f'ang:{angular:.2f} | [{self.action_label}]')


    def stop_robot(self):
        """Publishes zero velocity — stops the robot."""
        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)


    def publish_label(self):
        """Publishes current action label as a ROS topic."""
        msg = String()
        msg.data = self.action_label
        self.label_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopping wall follower...')
        node.stop_robot()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()