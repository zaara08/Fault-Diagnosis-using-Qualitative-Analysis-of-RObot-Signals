# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import LaserScan
# from geometry_msgs.msg import Twist
# from std_msgs.msg import String
# import math

# # ── TUNING PARAMETERS ──────────────────────────────────────────
# TARGET_DISTANCE  = 0.5   # metres from wall (left side)
# KP               = 1.2   # proportional gain (how strongly to correct)
# FORWARD_SPEED    = 0.1   # m/s forward speed (slow and safe)
# MAX_ANGULAR      = 0.5   # maximum turning speed (rad/s)
# WALL_FOUND_DIST  = 2.0   # if left side < this, wall is detected
# DANGER_FRONT     = 0.25  # emergency stop if front closer than this
# PILLAR_THRESHOLD = 0.35  # if wall suddenly < this, pillar detected
# # ───────────────────────────────────────────────────────────────


# class WallFollowerNode(Node):
#     """
#     Wall follower using a P-controller.
#     The robot keeps the wall on its LEFT side at TARGET_DISTANCE.
#     Works for both plain walls and walls with pillars.
#     """

#     def __init__(self):
#         super().__init__('wall_follower_node')

#         # --- Publisher: sends movement commands to robot ---
#         self.cmd_pub = self.create_publisher(
#             Twist, '/cmd_vel', 10)

#         # --- Publisher: sends action label to data logger ---
#         self.label_pub = self.create_publisher(
#             String, '/action_label', 10)

#         # --- Subscriber: receives LIDAR data ---
#         self.create_subscription(
#             LaserScan, '/scan', self.scan_callback, 10)

#         # --- Timer: runs control loop every 0.1 seconds ---
#         self.create_timer(0.1, self.control_loop)

#         # --- Internal state variables ---
#         self.lidar_right       = None   # distance to right wall
#         self.lidar_front_right = None   # distance front-right
#         self.lidar_front       = None   # distance straight ahead
#         self.action_label      = 'idle' # current action label

#         self.get_logger().info('Wall Follower Node started!')
#         self.get_logger().info(
#             f'Target distance: {TARGET_DISTANCE}m | '
#             f'Speed: {FORWARD_SPEED}m/s')


#     def scan_callback(self, msg):
#         """
#         Called every time a new LIDAR scan arrives.
#         Extracts 3 key distances from the full scan.

#         The LIDAR scans from -90deg (right) to +90deg (left).
#         We extract:
#           - right      (index for -90 deg)
#           - front_right (index for -45 deg)
#           - front      (index for   0 deg)
#         """
#         ranges = msg.ranges
#         total  = len(ranges)

#         # Helper to safely get a LIDAR value
#         # (some beams return 'inf' if nothing detected)
#         def safe_get(index):
#             val = ranges[index]
#             if math.isinf(val) or math.isnan(val):
#                 return msg.range_max
#             return val

#         # Extract the three key angles
#         # index 0   = rightmost (-90 deg)
#         # index mid = front (0 deg)
#         # index end = leftmost (+90 deg)
#         # right_idx       = 0
#         # front_right_idx = total // 4       # -45 degrees
#         # front_idx       = total // 2       #   0 degrees

#         # self.lidar_right       = safe_get(right_idx)
#         # self.lidar_front_right = safe_get(front_right_idx)
#         # self.lidar_front       = safe_get(front_idx)
#         # LEFT side wall following
#         # index 0   = right (-90 deg)
#         # index end = left  (+90 deg)
#         left_idx       = total - 1        # +90 degrees (left)
#         front_left_idx = (total * 3) // 4 # +45 degrees (front-left)
#         front_idx      = total // 2       #   0 degrees (front)

#         self.lidar_right       = safe_get(left_idx)
#         self.lidar_front_right = safe_get(front_left_idx)
#         self.lidar_front       = safe_get(front_idx)


#     def get_action_label(self, right, front):
#         """
#         Decides what the robot is currently doing
#         based on LIDAR readings.
#         Returns a string label for the CSV.
#         """
#         # Emergency: something very close in front
#         if front < DANGER_FRONT:
#             return 'wall_following_fail'

#         # No wall detected on right side yet
#         if right > WALL_FOUND_DIST:
#             return 'searching_wall'

#         # Approaching: robot is far from wall
#         if right > TARGET_DISTANCE + 0.15:
#             return 'approaching_wall'

#         # Pillar detected: wall suddenly very close
#         if right < PILLAR_THRESHOLD:
#             return 'wall_avoiding_pillar'

#         # Too close (but not pillar)
#         if right < TARGET_DISTANCE - 0.15:
#             return 'wall_following_fail'

#         # Perfect: robot is at target distance
#         return 'wall_following'


#     def control_loop(self):
#         """
#         Runs every 0.1 seconds.
#         Reads LIDAR values, calculates correction,
#         publishes movement command and action label.
#         """
#         # Wait until first LIDAR reading arrives
#         if self.lidar_right is None:
#             return

#         right = self.lidar_right
#         front = self.lidar_front
#         front_right = self.lidar_front_right

#         # ── SAFETY CHECK ───────────────────────────────────
#         # If something is dangerously close in front → STOP
#         if front < DANGER_FRONT:
#             self.stop_robot()
#             self.action_label = 'wall_following_fail'
#             self.publish_label()
#             self.get_logger().warn(
#                 f'EMERGENCY STOP! Front obstacle: {front:.2f}m')
#             return

#         # ── SEARCHING ──────────────────────────────────────
#         # No wall on right side → rotate slowly to find it
#         if right > WALL_FOUND_DIST:
#             cmd = Twist()
#             cmd.linear.x  = 0.0
#             # cmd.angular.z = -0.3   # rotate right to find wall
#             cmd.angular.z = 0.3   # rotate left to find left wall
#             self.cmd_pub.publish(cmd)
#             self.action_label = 'searching_wall'
#             self.publish_label()
#             return

#         # ── P-CONTROLLER ───────────────────────────────────
#         # error = how far we are from target distance
#         # positive error = too far from wall → turn right
#         # negative error = too close to wall → turn left
#         error = right - TARGET_DISTANCE

#         # Special case: pillar approaching (front_right close)
#         # Steer away more aggressively
#         if front_right < PILLAR_THRESHOLD:
#             error = -0.3   # force left turn to avoid pillar

#         # Calculate angular correction
#         angular = KP * error
#         # Clamp to maximum turning speed
#         angular = max(-MAX_ANGULAR, min(MAX_ANGULAR, angular))

#         # Build and send movement command
#         cmd = Twist()
#         cmd.linear.x  = FORWARD_SPEED
#         # cmd.angular.z = -angular   # negative = turn right
#         cmd.angular.z = angular    # positive = turn left (toward left wall)

#         self.cmd_pub.publish(cmd)

#         # ── ACTION LABEL ───────────────────────────────────
#         self.action_label = self.get_action_label(right, front)
#         self.publish_label()

#         # ── LOGGING ────────────────────────────────────────
#         self.get_logger().info(
#             f'R:{right:.2f}m FR:{front_right:.2f}m '
#             f'F:{front:.2f}m | err:{error:.2f} '
#             f'ang:{angular:.2f} | [{self.action_label}]')


#     def stop_robot(self):
#         """Publishes zero velocity — stops the robot."""
#         cmd = Twist()
#         cmd.linear.x  = 0.0
#         cmd.angular.z = 0.0
#         self.cmd_pub.publish(cmd)


#     def publish_label(self):
#         """Publishes current action label as a ROS topic."""
#         msg = String()
#         msg.data = self.action_label
#         self.label_pub.publish(msg)


# def main(args=None):
#     rclpy.init(args=args)
#     node = WallFollowerNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('Stopping wall follower...')
#         node.stop_robot()
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math

# ── TUNING PARAMETERS ──────────────────────────────────────────
TARGET_DISTANCE  = 0.5   # metres from wall (LEFT side)
PILLAR_THRESHOLD = 0.45  # fail if closer than this
KP               = 0.8   # proportional gain for sideways correction
FORWARD_SPEED    = 0.1   # m/s forward speed (slow and safe)
MAX_SIDEWAYS     = 0.3   # maximum sideways speed (m/s)
WALL_FOUND_DIST  = 2.0   # wall detected if closer than this
DANGER_FRONT     = 0.25  # emergency stop if front closer than this
STOP_DISTANCE    = 4.12  # stop after travelling this far (metres)
# ───────────────────────────────────────────────────────────────


class WallFollowerNode(Node):
    """
    Holonomic wall follower using a P-controller.
    The robot keeps the wall on its LEFT side at TARGET_DISTANCE.
    Uses sideways motion instead of rotation — no spinning!
    Stops automatically after STOP_DISTANCE metres.
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
        self.lidar_left        = None  # distance to left wall
        self.lidar_front_left  = None  # distance front-left
        self.lidar_front       = None  # distance straight ahead
        self.action_label      = 'idle'

        # --- Distance tracking ---
        self.start_x           = None  # starting x position
        self.start_y           = None  # starting y position
        self.current_x         = 0.0
        self.current_y         = 0.0
        self.distance_travelled = 0.0
        self.stopped           = False

        # --- Odometry subscriber ---
        from nav_msgs.msg import Odometry
        self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.get_logger().info('Holonomic Wall Follower started!')
        self.get_logger().info(
            f'Target: {TARGET_DISTANCE}m from LEFT wall | '
            f'Stop after: {STOP_DISTANCE}m')


    def scan_callback(self, msg):
        """Extracts LEFT side LIDAR distances."""
        ranges = msg.ranges
        total  = len(ranges)

        def safe_get(index):
            val = ranges[index]
            if math.isinf(val) or math.isnan(val):
                return round(msg.range_max, 3)
            return round(val, 3)

        # LEFT side indices
        self.lidar_left       = safe_get(total - 1)      # +90 deg (left)
        self.lidar_front_left = safe_get((total * 3)//4) # +45 deg (front-left)
        self.lidar_front      = safe_get(total // 2)     #   0 deg (front)


    def odom_callback(self, msg):
        """Tracks robot position to measure distance travelled."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Record starting position on first reading
        if self.start_x is None:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.get_logger().info(
                f'Start position recorded: '
                f'({self.start_x:.2f}, {self.start_y:.2f})')

        # Calculate distance travelled from start
        dx = self.current_x - self.start_x
        dy = self.current_y - self.start_y
        self.distance_travelled = math.sqrt(dx*dx + dy*dy)


    def get_action_label(self, left, front):
        """Decides current action label for CSV."""
        if front < DANGER_FRONT:
            return 'wall_following_fail'
        if left > WALL_FOUND_DIST:
            return 'searching_wall'
        if left > TARGET_DISTANCE + 0.15:
            return 'approaching_wall'
        if left < TARGET_DISTANCE - 0.10:
            return 'wall_following_fail'
        return 'wall_following'


    def stop_robot(self):
        """Sends zero velocity to stop robot completely."""
        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.linear.y  = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)


    def publish_label(self):
        """Publishes action label to data logger."""
        msg = String()
        msg.data = self.action_label
        self.label_pub.publish(msg)


    def control_loop(self):
        """
        Runs every 0.1 seconds.
        Uses HOLONOMIC motion — no rotation!
        linear.x = forward speed (always constant)
        linear.y = sideways correction (toward/away from wall)
        angular.z = always 0 (no spinning!)
        """
        # Wait for first LIDAR reading
        if self.lidar_left is None:
            return

        # ── AUTO STOP ──────────────────────────────────────
        # Stop automatically after STOP_DISTANCE metres
        if self.distance_travelled >= STOP_DISTANCE:
            if not self.stopped:
                self.stop_robot()
                self.stopped = True
                self.action_label = 'idle'
                self.publish_label()
                self.get_logger().info(
                    f'REACHED END! Distance: '
                    f'{self.distance_travelled:.2f}m — Stopping!')
            return

        left  = self.lidar_left
        front = self.lidar_front

        # ── SAFETY CHECK ───────────────────────────────────
        if front < DANGER_FRONT:
            self.stop_robot()
            self.action_label = 'wall_following_fail'
            self.publish_label()
            self.get_logger().warn(
                f'EMERGENCY STOP! Front: {front:.2f}m')
            return

        # ── NO WALL FOUND ──────────────────────────────────
        # Move sideways LEFT to find wall
        # No rotation — just slide left!
        if left > WALL_FOUND_DIST:
            cmd = Twist()
            cmd.linear.x  = 0.0
            cmd.linear.y  = 0.2   # slide LEFT to find wall
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.action_label = 'searching_wall'
            self.publish_label()
            self.get_logger().info(
                f'Searching... sliding left | '
                f'L:{left:.2f}m | '
                f'Dist:{self.distance_travelled:.2f}m')
            return

        # ── P-CONTROLLER (HOLONOMIC) ────────────────────────
        # error = how far we are from target distance
        # positive = too far from wall → move LEFT (toward wall)
        # negative = too close to wall → move RIGHT (away from wall)
        error = left - TARGET_DISTANCE

        # Calculate sideways correction
        sideways = KP * error
        # Clamp to maximum sideways speed
        sideways = max(-MAX_SIDEWAYS, min(MAX_SIDEWAYS, sideways))

        # Build movement command
        cmd = Twist()
        cmd.linear.x  = FORWARD_SPEED  # always move forward
        cmd.linear.y  = sideways        # sideways correction
        cmd.angular.z = 0.0             # NO rotation!

        self.cmd_pub.publish(cmd)

        # ── ACTION LABEL ───────────────────────────────────
        self.action_label = self.get_action_label(left, front)
        self.publish_label()

        # ── LOGGING ────────────────────────────────────────
        self.get_logger().info(
            f'L:{left:.2f}m F:{front:.2f}m | '
            f'err:{error:.2f} side:{sideways:.2f} | '
            f'dist:{self.distance_travelled:.2f}m | '
            f'[{self.action_label}]')


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