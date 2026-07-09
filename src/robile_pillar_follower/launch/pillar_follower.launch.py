#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Starts both pillar follower nodes together.
    """

    pillar_follower_node = Node(
        package='robile_pillar_follower',
        executable='pillar_follower_node',
        name='pillar_follower_node',
        output='screen',
    )

    data_logger_node = Node(
        package='robile_pillar_follower',
        executable='data_logger_node',
        name='data_logger_node',
        output='screen',
    )

    return LaunchDescription([
        pillar_follower_node,
        data_logger_node,
    ])