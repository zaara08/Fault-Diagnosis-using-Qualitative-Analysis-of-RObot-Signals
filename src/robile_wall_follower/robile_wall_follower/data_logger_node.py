#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import csv
import os
import math
from datetime import datetime


class DataLoggerNode(Node):
    """
    Collects data from:
    - /scan      (LIDAR distances)
    - /odom      (position + speed)
    - /cmd_vel   (movement commands)
    - /imu       (rotation + acceleration)
    - /joint_states (wheel speeds)
    Saves everything to CSV every 0.1 seconds.
    """

    def __init__(self):
        super().__init__('data_logger_node')

        # ── CREATE CSV FILE ────────────────────────────────
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        home_dir  = os.path.expanduser('~')
        data_dir  = os.path.join(home_dir, 'rnd_ws', 'data')
        os.makedirs(data_dir, exist_ok=True)

        filename = os.path.join(
            data_dir, f'wall_data_{timestamp}.csv')

        self.csv_file   = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Write header row with ALL columns
        self.csv_writer.writerow([
            # Time
            'timestamp',
            # LIDAR
            'lidar_right',
            'lidar_front_right',
            'lidar_front',
            'lidar_left',
            # Odometry
            'odom_linear_x',
            'odom_angular_z',
            'pos_x',
            'pos_y',
            # Commands sent to robot
            'cmd_linear_x',
            'cmd_angular_z',
            # IMU
            'imu_angular_z',
            'imu_accel_x',
            'imu_accel_y',
            # Wheel speeds
            'wheel_speed_0',
            'wheel_speed_1',
            'wheel_speed_2',
            'wheel_speed_3',
            # Action label
            'action_label'
        ])

        self.get_logger().info(f'Saving data to: {filename}')

        # ── INTERNAL STATE ─────────────────────────────────
        # LIDAR
        self.lidar_right       = 0.0
        self.lidar_front_right = 0.0
        self.lidar_front       = 0.0
        self.lidar_left        = 0.0
        # Odometry
        self.odom_linear_x     = 0.0
        self.odom_angular_z    = 0.0
        self.pos_x             = 0.0
        self.pos_y             = 0.0
        # Commands
        self.cmd_linear_x      = 0.0
        self.cmd_angular_z     = 0.0
        # IMU
        self.imu_angular_z     = 0.0
        self.imu_accel_x       = 0.0
        self.imu_accel_y       = 0.0
        # Wheel speeds
        self.wheel_speeds      = [0.0, 0.0, 0.0, 0.0]
        # Label
        self.action_label      = 'idle'
        self.row_count         = 0

        # ── SUBSCRIBERS ────────────────────────────────────
        self.create_subscription(
            LaserScan, '/scan',
            self.scan_callback, 10)

        self.create_subscription(
            Odometry, '/odom',
            self.odom_callback, 10)

        self.create_subscription(
            Twist, '/cmd_vel',
            self.cmd_callback, 10)

        self.create_subscription(
            Imu, '/imu',
            self.imu_callback, 10)

        self.create_subscription(
            JointState, '/joint_states',
            self.joint_callback, 10)

        self.create_subscription(
            String, '/action_label',
            self.label_callback, 10)

        # Save one row every 0.1 seconds
        self.create_timer(0.1, self.save_row)

        self.get_logger().info('Data Logger Node started!')
        self.get_logger().info(
            'Collecting: LIDAR + ODOM + CMD + IMU + WHEELS')


    def scan_callback(self, msg):
        """Extracts 4 key LIDAR distances."""
        ranges = msg.ranges
        total  = len(ranges)

        def safe_get(index):
            val = ranges[index]
            if math.isinf(val) or math.isnan(val):
                return round(msg.range_max, 3)
            return round(val, 3)

        self.lidar_right       = safe_get(0)
        self.lidar_front_right = safe_get(total // 4)
        self.lidar_front       = safe_get(total // 2)
        self.lidar_left        = safe_get(total - 1)


    def odom_callback(self, msg):
        """Records robot speed and position."""
        self.odom_linear_x  = round(
            msg.twist.twist.linear.x, 4)
        self.odom_angular_z = round(
            msg.twist.twist.angular.z, 4)
        self.pos_x = round(
            msg.pose.pose.position.x, 4)
        self.pos_y = round(
            msg.pose.pose.position.y, 4)


    def cmd_callback(self, msg):
        """Records movement commands sent to robot."""
        self.cmd_linear_x  = round(msg.linear.x, 4)
        self.cmd_angular_z = round(msg.angular.z, 4)


    def imu_callback(self, msg):
        """Records IMU rotation and acceleration."""
        self.imu_angular_z = round(
            msg.angular_velocity.z, 4)
        self.imu_accel_x   = round(
            msg.linear_acceleration.x, 4)
        self.imu_accel_y   = round(
            msg.linear_acceleration.y, 4)


    def joint_callback(self, msg):
        """Records individual wheel speeds."""
        # Joint states has velocities for each wheel
        velocities = msg.velocity
        for i in range(min(4, len(velocities))):
            self.wheel_speeds[i] = round(velocities[i], 4)


    def label_callback(self, msg):
        """Receives action label from wall follower."""
        self.action_label = msg.data


    def save_row(self):
        """Writes one complete row to CSV every 0.1 seconds."""
        timestamp = datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S.%f')[:-3]

        self.csv_writer.writerow([
            timestamp,
            # LIDAR
            self.lidar_right,
            self.lidar_front_right,
            self.lidar_front,
            self.lidar_left,
            # Odometry
            self.odom_linear_x,
            self.odom_angular_z,
            self.pos_x,
            self.pos_y,
            # Commands
            self.cmd_linear_x,
            self.cmd_angular_z,
            # IMU
            self.imu_angular_z,
            self.imu_accel_x,
            self.imu_accel_y,
            # Wheel speeds
            self.wheel_speeds[0],
            self.wheel_speeds[1],
            self.wheel_speeds[2],
            self.wheel_speeds[3],
            # Label
            self.action_label
        ])

        # Flush every 10 rows to protect data
        self.row_count += 1
        if self.row_count % 10 == 0:
            self.csv_file.flush()
            self.get_logger().info(
                f'Rows: {self.row_count} | '
                f'R:{self.lidar_right:.2f}m | '
                f'Label: [{self.action_label}]')


    def destroy_node(self):
        """Closes CSV file cleanly on shutdown."""
        self.csv_file.flush()
        self.csv_file.close()
        self.get_logger().info(
            f'Data saved! Total rows: {self.row_count}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopping data logger...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()