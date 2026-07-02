#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Starts both nodes together with one command.
    Like pressing one ON button for everything.
    """

    wall_follower_node = Node(
        package='robile_wall_follower',
        executable='wall_follower_node',
        name='wall_follower_node',
        output='screen',   # shows logs in terminal
    )

    data_logger_node = Node(
        package='robile_wall_follower',
        executable='data_logger_node',
        name='data_logger_node',
        output='screen',   # shows logs in terminal
    )

    return LaunchDescription([
        wall_follower_node,
        data_logger_node,
    ])