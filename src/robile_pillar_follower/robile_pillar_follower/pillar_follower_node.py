#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math

# ── TUNING PARAMETERS ──────────────────────────────────────────
TARGET_DISTANCE   = 0.5   # metres from wall (LEFT side)
KP                = 0.8   # proportional gain
FORWARD_SPEED     = 0.1   # m/s forward speed
MAX_SIDEWAYS      = 0.3   # maximum sideways speed
WALL_FOUND_DIST   = 2.0   # wall detected if closer than this
DANGER_FRONT      = 0.30  # emergency stop if front closer than this
PILLAR_DETECT     = 0.38  # lidar_left drops below this = pillar!
PILLAR_DEPTH      = 0.17  # pillar sticks out 17cm
STOP_TIME         = 75.0  # stop after 75 seconds
AVOIDANCE_OFFSET  = 0.35  # move this much away from wall
DETECTION_STEPS   = 5     # show obstacle_detected for 5 steps
# ───────────────────────────────────────────────────────────────


class PillarFollowerNode(Node):
    """
    Holonomic wall follower with pillar avoidance.

    Action Labels:
    - idle                : startup or stopped
    - searching_wall      : looking for wall
    - approaching_wall    : moving toward wall
    - wall_following      : stable at 0.5m ✅
    - obstacle_detected   : pillar seen ⚠️
    - obstacle_avoidance  : moving around pillar 🔄
    - wall_reacquired     : found wall after pillar 🔄
    - wall_following_fail : crashed or too close ❌
    """

    def __init__(self):
        super().__init__('pillar_follower_node')

        # --- Publishers ---
        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        self.label_pub = self.create_publisher(
            String, '/action_label', 10)

        # --- Subscribers ---
        self.create_subscription(
            LaserScan, '/scan',
            self.scan_callback, 10)

        from nav_msgs.msg import Odometry
        self.create_subscription(
            Odometry, '/odom',
            self.odom_callback, 10)

        # --- Timer ---
        self.create_timer(0.1, self.control_loop)

        # --- LIDAR values ---
        self.lidar_left       = None
        self.lidar_front_left = None
        self.lidar_front      = None
        self.lidar_right      = None

        # --- State machine ---
        self.action_label     = 'idle'

        # --- Detection state ---
        self.detecting        = False
        self.detected_count   = 0

        # --- Avoidance state ---
        self.avoiding         = False
        self.avoidance_count  = 0
        self.AVOIDANCE_STEPS  = 25  # 2.5 seconds

        # --- Recovery state ---
        self.recovering       = False
        self.recovery_count   = 0
        self.RECOVERY_STEPS   = 15  # 1.5 seconds

        # --- Timer based stop ---
        self.start_time       = None
        self.stopped          = False

        # --- Distance tracking ---
        self.start_x          = None
        self.start_y          = None
        self.current_x        = 0.0
        self.current_y        = 0.0
        self.distance_travelled = 0.0

        self.get_logger().info('Pillar Follower Node started!')
        self.get_logger().info(
            f'Target: {TARGET_DISTANCE}m | '
            f'Stop after: {STOP_TIME}s | '
            f'Pillar detect: {PILLAR_DETECT}m | '
            f'Detection steps: {DETECTION_STEPS}')


    def scan_callback(self, msg):
        """Extracts LIDAR distances from all key angles."""
        ranges = msg.ranges
        total  = len(ranges)

        def safe_get(index):
            val = ranges[index]
            if math.isinf(val) or math.isnan(val):
                return round(msg.range_max, 3)
            return round(val, 3)

        self.lidar_right      = safe_get(0)
        self.lidar_front      = safe_get(total // 2)
        self.lidar_front_left = safe_get((total * 3) // 4)
        self.lidar_left       = safe_get(total - 1)


    def odom_callback(self, msg):
        """Tracks distance travelled."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        if self.start_x is None:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.get_logger().info(
                f'Start recorded: '
                f'({self.start_x:.2f}, {self.start_y:.2f})')

        dx = self.current_x - self.start_x
        dy = self.current_y - self.start_y
        self.distance_travelled = math.sqrt(dx*dx + dy*dy)


    def stop_robot(self):
        """Sends zero velocity."""
        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.linear.y  = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)


    def publish_label(self, label):
        """Publishes action label."""
        self.action_label = label
        msg = String()
        msg.data = label
        self.label_pub.publish(msg)


    def control_loop(self):
        """
        Main control loop — runs every 0.1 seconds.

        State machine:
        searching → approaching → wall_following
                                      ↓
                              obstacle_detected (5 steps)
                                      ↓
                              obstacle_avoidance (25 steps)
                                      ↓
                              wall_reacquired (15 steps)
                                      ↓
                              wall_following (again!)
                                      ↓
                              idle (after 75 seconds)
        """
        if self.lidar_left is None:
            return

        left       = self.lidar_left
        front      = self.lidar_front
        front_left = self.lidar_front_left

        # ── TIMER BASED AUTO STOP ──────────────────────────────
        if self.start_time is None:
            self.start_time = self.get_clock().now()
            self.get_logger().info('Timer started!')

        elapsed = (
            self.get_clock().now() -
            self.start_time).nanoseconds / 1e9

        if elapsed >= STOP_TIME:
            if not self.stopped:
                self.stop_robot()
                self.stopped = True
                self.publish_label('idle')
                self.get_logger().info(
                    f'REACHED END! '
                    f'Time: {elapsed:.1f}s | '
                    f'Dist: {self.distance_travelled:.2f}m')
            return

        # ── EMERGENCY STOP ─────────────────────────────────────
        if (front < DANGER_FRONT and
                not self.avoiding and
                not self.detecting):
            self.stop_robot()
            self.publish_label('wall_following_fail')
            self.get_logger().warn(
                f'EMERGENCY STOP! Front: {front:.2f}m')
            return

        # ── STATE: SEARCHING ───────────────────────────────────
        if (left > WALL_FOUND_DIST and
                not self.avoiding and
                not self.detecting):
            cmd = Twist()
            cmd.linear.x  = 0.0
            cmd.linear.y  = 0.2
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.publish_label('searching_wall')
            self.get_logger().info(
                f'Searching wall... L:{left:.2f}m')
            return

        # ── STATE: OBSTACLE DETECTION ──────────────────────────
        # Pillar detected when lidar_left suddenly drops!
        if (not self.detecting and
                not self.avoiding and
                not self.recovering and
                left < PILLAR_DETECT):

            self.detecting      = True
            self.detected_count = 0
            self.get_logger().warn(
                f'PILLAR DETECTED! L:{left:.2f}m')

        # Show obstacle_detected label for DETECTION_STEPS
        if self.detecting:
            self.detected_count += 1

            # Slow down but keep moving during detection
            cmd = Twist()
            cmd.linear.x  = FORWARD_SPEED * 0.5
            cmd.linear.y  = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.publish_label('obstacle_detected')

            self.get_logger().warn(
                f'Obstacle detected! '
                f'step {self.detected_count}/'
                f'{DETECTION_STEPS} | '
                f'L:{left:.2f}m')

            # After detection steps → start avoidance
            if self.detected_count >= DETECTION_STEPS:
                self.detecting    = False
                self.avoiding     = True
                self.avoidance_count = 0
                self.get_logger().warn(
                    'Starting avoidance maneuver!')
            return

        # ── STATE: OBSTACLE AVOIDANCE ──────────────────────────
        if self.avoiding:
            self.avoidance_count += 1

            cmd = Twist()
            cmd.linear.x  = FORWARD_SPEED * 0.8
            cmd.linear.y  = -AVOIDANCE_OFFSET
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.publish_label('obstacle_avoidance')

            self.get_logger().info(
                f'Avoiding pillar... '
                f'step {self.avoidance_count}/'
                f'{self.AVOIDANCE_STEPS} | '
                f'L:{left:.2f}m | '
                f't:{elapsed:.1f}s')

            if self.avoidance_count >= self.AVOIDANCE_STEPS:
                self.avoiding       = False
                self.recovering     = True
                self.recovery_count = 0
                self.get_logger().info(
                    'Avoidance done! Starting recovery...')
            return

        # ── STATE: WALL RECOVERY ───────────────────────────────
        if self.recovering:
            self.recovery_count += 1

            cmd = Twist()
            cmd.linear.x  = FORWARD_SPEED * 0.8
            cmd.linear.y  = AVOIDANCE_OFFSET * 0.7
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.publish_label('wall_reacquired')

            self.get_logger().info(
                f'Recovering to wall... '
                f'step {self.recovery_count}/'
                f'{self.RECOVERY_STEPS} | '
                f'L:{left:.2f}m')

            if self.recovery_count >= self.RECOVERY_STEPS:
                self.recovering = False
                self.get_logger().info(
                    'Recovery done! Back to wall following!')
            return

        # ── STATE: P-CONTROLLER (NORMAL FOLLOWING) ─────────────
        error    = left - TARGET_DISTANCE
        sideways = KP * error
        sideways = max(-MAX_SIDEWAYS, min(MAX_SIDEWAYS, sideways))

        cmd = Twist()
        cmd.linear.x  = FORWARD_SPEED
        cmd.linear.y  = sideways
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

        # Determine label
        if left > TARGET_DISTANCE + 0.15:
            label = 'approaching_wall'
        elif left < TARGET_DISTANCE - 0.15:
            label = 'wall_following_fail'
        else:
            label = 'wall_following'

        self.publish_label(label)

        self.get_logger().info(
            f'L:{left:.2f}m FL:{front_left:.2f}m '
            f'F:{front:.2f}m | '
            f'err:{error:.2f} side:{sideways:.2f} | '
            f'dist:{self.distance_travelled:.2f}m | '
            f't:{elapsed:.1f}s | '
            f'[{self.action_label}]')


def main(args=None):
    rclpy.init(args=args)
    node = PillarFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopping...')
        node.stop_robot()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()