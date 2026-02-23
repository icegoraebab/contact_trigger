from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('contact_trigger'),
        'config',
        'reflex_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='contact_trigger',
            executable='reflex_grasp_node',
            name='reflex_grasp_node',
            output='screen',
            parameters=[config],
        )
    ])