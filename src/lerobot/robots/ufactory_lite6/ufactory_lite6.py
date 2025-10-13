import logging
import signal
from functools import cached_property
import time
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.constants import OBS_STATE
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.ufactory import XarmMotorBus
from lerobot.robots.robot import Robot
from lerobot.robots.robot_utils import ensure_safe_goal_position
from lerobot.robots.ufactory_lite6.config_ufactory_lite6 import UFactoryLite6Config


logger = logging.getLogger(__name__)


class UFactoryLite6(Robot):
    """
    uFactory Lite6 6DOF robotic arm integration for LeRobot.
    Uses XarmMotorBus for better control and safety features.
    """

    config_class = UFactoryLite6Config
    name = "ufactory_lite6"  

    def __init__(self, config: UFactoryLite6Config):
        super().__init__(config)
        self.config = config
        
        # Initialize XarmMotorBus for motor control (6 joints only, no gripper)
        self.bus = XarmMotorBus(
            port=config.ip,
            motors={
                "joint1": Motor(1, "ufactory_lite6-1", MotorNormMode.RANGE_0_100),
                "joint2": Motor(2, "ufactory_lite6-2", MotorNormMode.RANGE_0_100), 
                "joint3": Motor(3, "ufactory_lite6-3", MotorNormMode.RANGE_0_100),
                "joint4": Motor(4, "ufactory_lite6-4", MotorNormMode.RANGE_0_100),
                "joint5": Motor(5, "ufactory_lite6-5", MotorNormMode.RANGE_0_100),
                "joint6": Motor(6, "ufactory_lite6-6", MotorNormMode.RANGE_0_100),
            },
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "joint1.pos",
                "joint2.pos",
                "joint3.pos",
                "joint4.pos",
                "joint5.pos",
                "joint6.pos",
            ),
            float,
        )

   
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Support both position and velocity control
        features = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float  # Position control
            features[f"{motor}.vel"] = float  # Velocity control
        return features

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())
    
    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        # if not self.is_calibrated and calibrate:
        #     self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")
    
    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated
    
    def calibrate(self) -> None:
        return NotImplementedError

    def configure(self):
        """Configure the robot after connection."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        logger.info(f"Configuring {self}...")
        _ = self.bus.enable(follower=True, return_init_pos=True)
        
        # Set control mode based on configuration AFTER enable() call
        # This ensures our mode setting is not overridden by enable()
        if self.config.control_mode == "velocity":
            logger.info("Setting robot to velocity control mode...")
            self.bus.api.clean_error()
            self.bus.api.set_mode(4)  # Velocity control mode
            self.bus.api.set_state(0)
            self.bus.current_mode = 4  # Update tracked mode
            time.sleep(0.5)  # Give time for mode change
            logger.info("✅ Velocity control mode set successfully")
        else:
            logger.info("Setting robot to position control mode...")
            self.bus.api.clean_error()
            self.bus.api.set_mode(1)  # Position control mode
            self.bus.api.set_state(0)
            self.bus.current_mode = 1  # Update tracked mode
            time.sleep(0.5)  # Give time for mode change
            logger.info("✅ Position control mode set successfully")

    def get_observation(self) -> dict[str, Any]:
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.get_position()
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict
    
    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )
        
        if self.config.control_mode == "velocity":
            # Velocity control mode
            if any(key.endswith(".pos") for key in action.keys()):
                # Convert position to velocity for velocity control
                goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
                current_pos = self.bus.get_position()
                velocity = {}
                for motor, target_pos in goal_pos.items():
                    current_angle = current_pos.get(motor, 0.0)
                    # Calculate velocity with appropriate scaling for responsive control
                    # Scale by control frequency to make movement more responsive
                    vel = (target_pos - current_angle) * self.config.control_frequency * 0.1
                    velocity[motor] = vel
                self.bus.set_velocity(velocity)
                return {f"{motor}.vel": val for motor, val in velocity.items()}
            elif any(key.endswith(".vel") for key in action.keys()):
                # Direct velocity control
                velocity = {key.removesuffix(".vel"): val for key, val in action.items() if key.endswith(".vel")}
                self.bus.set_velocity(velocity)
                return {f"{motor}.vel": val for motor, val in velocity.items()}
            else:
                raise ValueError("Velocity control mode requires either '.pos' or '.vel' keys in action")
        elif self.config.control_mode == "position":
            # Position control mode (original behavior)
            goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
            
            # Cap goal position when too far away from present position.
            # /!\ Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.bus.get_position()
                goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
                goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

            # Send goal position to the arm
            self.bus.set_position(goal_pos)
            return {f"{motor}.pos": val for motor, val in goal_pos.items()}
        else:
            raise ValueError(f"Unsupported control mode: {self.config.control_mode}")
    
    def get_ee_coordinates(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        # Read arm position
        start = time.perf_counter()
        ee_pos = self.bus.get_cartesian_position()
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read ee coordinates: {dt_ms:.1f}ms")
        return ee_pos

    
    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")