
# Copyright (c) 2019 Andreas Klintberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # noqa
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'launch'))  # noqa

import launch
from launch import LaunchDescription
from launch import LaunchIntrospector
from launch import LaunchService

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from launch_ros import get_default_launch_description
import launch_ros.actions

p = Path(os.path.realpath(__file__))
PATH = "/".join(list(p.parts[1:-2]))


NAMESPACE = '/camera'

def generate_launch_description():
    ld = LaunchDescription()

    ################################
    # Drivers
    ################################
    usb_camera = launch_ros.actions.ComposableNodeContainer(
        node_name="usb_camera_driver_container",
        package='rclcpp_components',
        node_namespace=NAMESPACE,
        node_executable='component_container', 
        composable_node_descriptions=[
            ComposableNode(
                    package='usb_camera_driver',
                    node_plugin='usb_camera_driver::CameraDriver',
                    node_name='usb_camera_driver_node',
                    parameters=[
                        {"camera_calibration_file": "file:///<file-path>"}
                    ])
                ],
        output='screen'
        )

    ld.add_action(usb_camera)

    return ld


def main(argv=sys.argv[1:]):
    """Main."""
    ld = generate_launch_description()

    print('Starting introspection of launch description...')
    print('')

    print(LaunchIntrospector().format_launch_description(ld))

    print('')
    print('Starting launch of launch description...')
    print('')

    ls = LaunchService()
    ls.include_launch_description(get_default_launch_description())
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == '__main__':
    main()
