#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


# ufactory lite6 config
@RobotConfig.register_subclass("ufactory_lite6")
@dataclass
class UFactoryLite6Config(RobotConfig):
    """Configuration for UFactory Lite6 6DOF robot arm."""
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.

    # Robot IP address (default is ufactory lite6 ip)
    ip: str = "192.168.1.193"  
    
    # Camera configurations 
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # cameras: dict[str, CameraConfig] = field(
    #     default_factory=lambda: {
    #         "webcam_1": OpenCVCameraConfig(
    #             index_or_path=6,
    #             fps=30,
    #             width=640,
    #             height=480,
    #         ),
    #         "webcam_2": OpenCVCameraConfig(
    #             index_or_path=6,
    #             fps=30,
    #             width=640,
    #             height=480,
    #         ),
    #     }
    # )
    mock: bool = False

    robot_type: str = "ufactory_lite6"

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None


    # Control parameters
    control_frequency: float = 10.0  # Hz， ufactory lite6 control frequency
    control_mode: str = "velocity"  # Default to position control, can be changed to "velocity"
    use_gripper: bool = False  # 6DOF version doesn't have gripper
    use_degrees: bool = True  # if true, the position is in degrees, otherwise in radians
    


