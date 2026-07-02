#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import csv
import os
import math
from datetime import datetime


class DataLoggerNode(Node):
    """
    Listens to LIDAR, odometry, cmd_vel, and action_label topics.
    Saves everything to a CSV file every 0.1 seconds.
    Each run creates a new timestamped CSV file automatically.
    """

    def __init__(self):
        super().__init__('data_logger_node')

        # ── CREATE CSV FILE ────────────────────────────────
        # Each run gets its own file with timestamp in name
        # Example: wall_data_2026-07-02_14-30-00.csv
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        home_dir  = os.path.expanduser('~')
        data_dir  = os.path.join(home_dir, 'rnd_ws', 'data')

        # Create data folder if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        filename = os.path.join(
            data_dir, f'wall_data_{timestamp}.csv')

        # Open CSV file and write header row
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp',
            'lidar_right',
            'lidar_front_right',
            'lidar_front',
            'lidar_left',
            'linear_x',
            'angular_z',
            'pos_x',
            'pos_y',
            'action_label'
        ])

        self.get_logger().info(
            f'Saving data to: {filename}')

        # ── INTERNAL STATE ─────────────────────────────────
        self.lidar_right       = 0.0
        self.lidar_front_right = 0.0
        self.lidar_front       = 0.0
        self.lidar_left        = 0.0
        self.linear_x          = 0.0
        self.angular_z         = 0.0
        self.pos_x             = 0.0
        self.pos_y             = 0.0
        self.action_label      = 'idle'
        self.row_count         = 0

        # ── SUBSCRIBERS ────────────────────────────────────
        self.create_subscription(
            LaserScan, '/scan',
            self.scan_callback, 10)

        self.create_subscription(
            Twist, '/cmd_vel',
            self.cmd_callback, 10)

        self.create_subscription(
            Odometry, '/odom',
            self.odom_callback, 10)

        self.create_subscription(
            String, '/action_label',
            self.label_callback, 10)

        # ── TIMER: save one row every 0.1 seconds ──────────
        self.create_timer(0.1, self.save_row)

        self.get_logger().info('Data Logger Node started!')


    def scan_callback(self, msg):
        """Extracts 4 key LIDAR distances from full scan."""
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


    def cmd_callback(self, msg):
        """Records the movement commands sent to robot."""
        self.linear_x  = round(msg.linear.x, 4)
        self.angular_z = round(msg.angular.z, 4)


    def odom_callback(self, msg):
        """Records robot position from odometry."""
        self.pos_x = round(
            msg.pose.pose.position.x, 4)
        self.pos_y = round(
            msg.pose.pose.position.y, 4)


    def label_callback(self, msg):
        """Receives action label from wall follower node."""
        self.action_label = msg.data


    def save_row(self):
        """
        Runs every 0.1 seconds.
        Writes one row to the CSV with all current values.
        """
        timestamp = datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S.%f')[:-3]

        self.csv_writer.writerow([
            timestamp,
            self.lidar_right,
            self.lidar_front_right,
            self.lidar_front,
            self.lidar_left,
            self.linear_x,
            self.angular_z,
            self.pos_x,
            self.pos_y,
            self.action_label
        ])

        # Flush every 10 rows so data is safe
        # even if program crashes
        self.row_count += 1
        if self.row_count % 10 == 0:
            self.csv_file.flush()
            self.get_logger().info(
                f'Rows saved: {self.row_count} | '
                f'Label: {self.action_label}')


    def destroy_node(self):
        """Called when node shuts down — closes CSV file cleanly."""
        self.csv_file.flush()
        self.csv_file.close()
        self.get_logger().info(
            f'CSV file closed. Total rows: {self.row_count}')
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