from copy import deepcopy
from enum import Enum
from xarm.wrapper import XArmAPI
import numpy as np
import threading
import time
import logging

from lerobot.utils.encoding_utils import decode_sign_magnitude, encode_sign_magnitude

from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorsBus, NameOrID, Value, get_address

class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0

class DriveMode(Enum):
    NON_INVERTED = 0
    INVERTED = 1

class XarmMotorBus(MotorsBus):
    """Wrapper for the xArm Python SDK"""

    def __init__(
        self,
        port: str,
        motors: dict[str, tuple[int, str]],
        mock=False,
    ):
        print("Initializing xArmWrapper")  # Debug print
        self.port = port
        self.motors = motors
        self.mock = mock

        self.calibration = None
        self._connected = False
        self.logs = {}

        self.api = None

        self.MAX_SPEED_LIMIT = None
        self.MAX_ACC_LIMIT = None

        # Stop event
        self.stop_event = threading.Event()
        self.replay_mode = False
        self.cartesian_mode = False
        self.current_mode = None  # Track current control mode

        # Create and start the digital input monitoring thread
        print("Creating monitor thread")  # Debug print
        self.monitor_input_thread = threading.Thread(target=self.monitor_digital_input, args=(self.stop_event,))


    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [motors.model for motors in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [motors.id for motors in self.motors.values()]

    @property
    def is_connected(self):
        return self._connected

    def connect(self):
        print("Connecting to xArm")  # Debug print
        print(self.port)
        if self._connected:
            print(f"DynamixelMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice.")
            #raise RobotDeviceAlreadyConnectedError(
                #f"DynamixelMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            #)

        if self.mock:
            print("Mock mode, not connecting to real device")  # Debug print
            return
        else:
            self.api = XArmAPI(self.port)

        try:
            if not self.api.connected:
                raise OSError(f"Failed to connect to xArm API @ '{self.port}'.")
            print("Successfully connected to xArm")  # Debug print
        except Exception as e:
            print(f"Exception while connecting in xArmWrapper: {e}")
            raise

        # Allow to read and write
        self._connected = True

        # Start the monitoring thread after successful connection
        self.monitor_input_thread.start()
        print("Monitor thread started")  # Debug print

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        pass  # TODO (@vmayoral): implement if of interest

    def read(self, data_name, motor_names: str | list[str] | None = None):
        pass  # TODO (@vmayoral): implement if of interest

    # Minimal API to be compatible with gym_manipulator reset_follower_position
    def sync_read(self, register: str, normalize: bool = False):
        if register == "Present_Position":
            return self.get_position()
        raise NotImplementedError(f"sync_read unsupported register: {register}")

    def sync_write(self, register: str, values: dict):
        if register == "Goal_Position":
            return self.set_position(values)
        raise NotImplementedError(f"sync_write unsupported register: {register}")

    def enable(self, follower: bool = False,return_init_pos: bool = False,init_pos: np.ndarray = None):
        self.api.motion_enable(enable=True)
        self.api.clean_error()
        self.api.set_mode(0)
        self.api.set_state(0)
        time.sleep(2)
        if init_pos is None:
            _, init_pos = tuple(self.api.get_initial_point())
        self.api.set_servo_angle(angle=init_pos,wait=True,is_radian=False)
        if follower:
            self.api.set_mode(1)
        else:
            self.api.set_mode(0)
        self.api.set_state(state=0)
        
        # Optional gripper setup (Lite6 no gripper, skip this)
        if self.motor_models and self.motor_models[-1] == "gripper":
            self.api.set_gripper_mode(0)
            self.api.set_gripper_enable(True)
            self.api.set_gripper_speed(5000)
        elif self.motor_models and self.motor_models[-1] == "robotiq":
            self.api.robotiq_reset()
            self.api.robotiq_set_activate()
        # # Initialize the global speed and acceleration limits
        # self.initialize_limits()  # not acting as expected
        # assume leader by default
        if not follower:
            self.api.set_mode(2)
            self.api.set_state(0)
            # Light up the digital output 2 (button), to signal manual mode
            self.api.set_tgpio_digital(ionum=2, value=1)
        
        if return_init_pos:
            # Return the initial position of the robot
            return np.array(init_pos)
        

    def disconnect(self):
        print("Disconnecting from xArm")  # Debug print
        if not self._connected:
            print(f"FeetechMotorsBus({self.port}) is not connected. Try running `motors_bus.connect()` first.")
            #raise RobotDeviceNotConnectedError(
            #    f"FeetechMotorsBus({self.port}) is not connected. Try running `motors_bus.connect()` first."
            #)

        # Turn off manual mode after recording
        self.api.set_mode(0)
        self.api.set_state(0)
        # Light down the digital output 2 (button), to signal manual mode
        self.api.set_tgpio_digital(ionum=2, value=0)
        # Disconnect both arms
        self.api.disconnect()

        # Stop events and threads
        self.stop_event.set()
        print("Waiting for monitor thread to join")  # Debug print
        self.monitor_input_thread.join()
        print("Monitor thread joined")  # Debug print

        # Signal as disconnected
        self._connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

    def initialize_limits(self):
        # heuristic: 1/3 of the max speed and acceleration limits
        #  for testing purposes
        self.MAX_SPEED_LIMIT = max(self.api.joint_speed_limit)/3
        self.MAX_ACC_LIMIT = max(self.api.joint_acc_limit)/3

    def get_position(self):
        # if self.motor_models[-1] == "gipper":
        #     code, angles = self.api.get_servo_angle()
        #     code_gripper, pos_gripper = self.api.get_gripper_position()
        #     pos = angles[:-1] + [pos_gripper]  # discard 7th dof, which is not present in U850
        #     #pos = angles + [pos_gripper]
        #     return pos
        # elif self.motor_models[-1] == "robotiq":
        #     code, angles = self.api.get_servo_angle()
        #     code_gripper, pos_gripper = self.api.robotiq_status['gFLT'], self.api.robotiq_status['gPO']
        #     pos = angles[:-1] + [pos_gripper]
        #     #pos = angles + [pos_gripper]
        #     return {motor: pos for motor, pos in zip(self.motor_names, pos)}
        code, angles = self.api.get_servo_angle()
        if code != 0:
            raise RuntimeError(f"Failed to get servo angles: {code}")
        # limit to the number of motors
        angles = angles[: len(self.motor_names)]
        return {motor: angle for motor, angle in zip(self.motor_names, angles)}
        
    def set_position_replay(self, position: np.ndarray):
        position = [i for _, i in list(position.items())]
        angles = position[:-1]
        gripper_pos = int(position[-1])

        if not self.replay_mode:
            self.api.set_mode(0)  # Set to follower mode
            self.api.set_state(0)  # Set to idle state
            self.replay_mode = True
        #angles = [round(angle,1) for angle in angles]  # Ensure angles are floats
        #angles = angles[:6]  # Ensure only the first 6 angles are used, as U850 has 6 joints + gripp

        
        # joints
        self.api.set_servo_angle(angle=angles, is_radian=False, wait=False, speed=self.MAX_SPEED_LIMIT, acceleration=self.MAX_ACC_LIMIT) 
        # gripper
        if self.motor_models[-1] == "gipper":
            self.api.set_gripper_position(pos=gripper_pos, wait=False)
        elif self.motor_models[-1] == "robotiq":
            self.api.robotiq_set_position(pos=gripper_pos)
        else:
            raise ValueError(f"Unsupported gripper model: {self.motor_models[-1]}")
        
    def get_cartesian_position(self):
        code, pos = self.api.get_position()
        if code != 0:
            raise RuntimeError(f"Failed to get Cartesian position: {code}")
        pos = np.array(pos)
        pos[:3] = pos[:3]/1000  # x, y, z in mm
        return pos

    def get_forward_kinematics(self):
        ## doesnt compute forward kinematics but gives 4x4 matrix of the end-effector
        position = self.get_cartesian_position()
        roll, pitch, yaw = position[3:]  # Extract orientation components
        # Convert degrees to radians
        roll, pitch, yaw = np.radians([roll, pitch, yaw])
    
        # Compute sines and cosines once
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        x, y, z = position[:3]  # Extract position components       
        # Create rotation matrix - RPY (Z-Y-X convention)
        R = np.array([
            [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, x],
            [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, y],
            [-sp, cp*sr, cp*cr, z],
            [0, 0, 0, 1]
        ])
        
        return R
    
    def compute_forward_kinematics(self, angles: np.ndarray):
        position = self.api.get_forward_kinematics(angles=angles)
        return np.array(position)
    
    def set_collision_sensing(self, mode: int):
        """
        Set the collision sensing mode.
        0: Disable collision sensing
        1: Enable collision sensing
        """
        if mode not in [0, 5]:
            raise ValueError("Invalid mode. Use 0 to disable or 1 to enable collision sensing.")
        self.api.set_collision_sensitivity(mode)

    def set_cartesian_position(self, position: np.ndarray):
        if not self.cartesian_mode:
            self.api.set_mode(7)  # Set to Cartesian mode
            self.api.set_state(0)  # Set to idle
            self.cartesian_mode = True
        x, y, z = position[:3]
        roll, pitch, yaw = position[3:]
        self.api.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, wait=False, speed=self.MAX_SPEED_LIMIT, acceleration=self.MAX_ACC_LIMIT)
    
    def _set_mode_if_needed(self, target_mode: int):
        """Set mode only if it's different from current mode to avoid warnings"""
        if self.current_mode != target_mode:
            self.api.clean_error()
            self.api.set_mode(target_mode)
            self.api.set_state(0)
            self.current_mode = target_mode
            time.sleep(0.1)  # Brief pause for mode change

    # position control mode
    def set_position(self,position: np.ndarray):
        # only joint angles, no gripper
        # ensure in joint servo mode 
        angles = [position[name] for name in self.motor_names if name in position]
        # ensure in joint servo mode 
        self._set_mode_if_needed(1)  # Position control mode
        self.api.set_servo_angle_j(angles=angles, is_radian=False, wait=False)
    
    # velocity control mode
    def set_velocity(self, velocity: dict):
        """
        Set joint velocities for velocity control mode.
        Args:
            velocity: dict with motor names as keys and velocity values as values
        """
        # Convert velocity dict to list in correct order
        velocities = [velocity.get(name, 0.0) for name in self.motor_names]

        # Ensure we're in velocity control mode (mode 4 for joint velocity control)
        # Use smart mode setting to avoid unnecessary mode changes
        self._set_mode_if_needed(4)  # Velocity control mode

        # Send velocity commands directly without excessive scaling
        # Based on official example: arm.vc_set_joint_velocity([-50, 0, 0, 0, 0, 0])
        # The API expects velocity values in degrees/second for each joint
        code = self.api.vc_set_joint_velocity(velocities, is_radian=False, is_sync=False, duration=0)
        if code != 0:
            raise RuntimeError(f"Velocity control command failed with code: {code}")
        
    def set_gripper_pos(self, position: int):
        if self.motor_models[-1] == "gipper":
            self.api.set_gripper_position(pos=position, wait=False)
        elif self.motor_models[-1] == "robotiq":
            self.api.robotiq_set_position(pos=position)
        else:
            raise ValueError(f"Unsupported gripper model: {self.motor_models[-1]}")

    def monitor_digital_input(self, stop_event):
        print("Starting monitor_digital_input")  # Debug print
        SINGLE_CLICK_TIME = 0.2
        DOUBLE_CLICK_TIME = 0.5
        LONG_CLICK_TIME = 1.0

        last_press_time = 0
        last_release_time = 0
        last_click_time = 0
        long_click_detected = False
        click_count = 0
        long_click_state = True  # starts in manual mode

        while not stop_event.is_set():
            try:
                if self.api is not None and self._connected:
                    code, value = self.api.get_tgpio_digital(ionum=2)
                    # print(f"Digital input read: code={code}, value={value}")  # Debug print
                    if code == 0:
                        current_time = time.time()
                        
                        if value == 1:  # Button pressed
                            if last_press_time == 0:
                                last_press_time = current_time
                            elif not long_click_detected and current_time - last_press_time >= LONG_CLICK_TIME:
                                print("Long click detected -> Switching manual mode")
                                long_click_detected = True
                                long_click_state = not long_click_state
                                if long_click_state:
                                    self.api.set_tgpio_digital(ionum=2, value=1)
                                    # manual mode
                                    self.api.clean_error()
                                    self.api.set_mode(2)
                                    self.api.set_state(0)
                                else:
                                    self.api.set_tgpio_digital(ionum=2, value=0)
                                    # disable manual mode
                                    self.api.clean_error()
                                    self.api.set_mode(0)
                                    self.api.set_state(0)
                        else:  # Button released
                            if last_press_time != 0:
                                press_duration = current_time - last_press_time

                                if not long_click_detected:
                                    if press_duration < SINGLE_CLICK_TIME:
                                        click_count += 1
                                        if click_count == 1:
                                            last_click_time = current_time
                                        elif click_count == 2:
                                            if current_time - last_click_time < DOUBLE_CLICK_TIME:
                                                if self.motor_models[-1] == "gipper":
                                                    print("Double click detected -> Open gripper")
                                                    self.api.set_gripper_position(pos=600, wait=False)  # Open gripper
                                                    click_count = 0
                                                elif self.motor_models[-1] == "robotiq":
                                                    print("Double click detected -> Open gripper")
                                                    self.api.robotiq_open()
                                                    click_count = 0
                                            else:
                                                if self.motor_models[-1] == "gipper":
                                                    print("Single click detected -> Close gripper")
                                                    self.api.set_gripper_position(pos=50, wait=False)  # Close gripper
                                                    click_count = 1
                                                elif self.motor_models[-1] == "robotiq":
                                                    print("Single click detected -> Close gripper")
                                                    self.api.robotiq_close()
                                                    click_count = 1
                                                last_click_time = current_time
                                    else:
                                        if click_count == 1 and self.motor_models[-1] == "gipper":
                                            print("Single click detected -> Close gripper")
                                            self.api.set_gripper_position(pos=50, wait=False)  # Close gripper
                                            click_count = 0
                                        elif click_count == 1 and self.motor_models[-1] == "robotiq":
                                            print("Single click detected -> Close gripper")
                                            self.api.robotiq_close()
                                            click_count = 0

                                last_release_time = current_time
                                last_press_time = 0
                                long_click_detected = False

                        # Reset click count if too much time has passed since last click
                        if click_count == 1 and current_time - last_click_time >= DOUBLE_CLICK_TIME and self.motor_models[-1] == "gipper":
                            print("Single click detected -> Close gripper")
                            self.api.set_gripper_position(pos=50, wait=False)  # Close gripper
                            click_count = 0
                        elif click_count == 1 and current_time - last_click_time >= DOUBLE_CLICK_TIME and self.motor_models[-1] == "robotiq":
                            print("Single click detected -> Close gripper")
                            self.api.robotiq_close()
                            click_count = 0

                else:
                    print("API not connected, waiting...")
                    time.sleep(1)  # Wait a bit longer before checking again
            except Exception as e:
                print(f"Error in monitor_digital_input: {e}")  # Debug print
            time.sleep(0.01)  # Check every 10ms for more precise detection
        print("Exiting monitor_digital_input")  # Debug print


    def robot_reset(self):
        if self.motor_models[-1] == "robotiq":
            self.api.robotiq_reset()
            raise ValueError(f"Unsupported gripper model: {self.motor_models[-1]}")
        elif self.motor_models[-1] == "gipper":
            self.api.set_gripper_position(pos=600, wait=True)
    
    def _assert_protocol_is_compatible(self, instruction_name):
        pass

    def _decode_sign(self, data_name, ids_values):
        pass

    def _disable_torque(self, motor, model, num_retry = 0):
        pass

    def _encode_sign(self, data_name, ids_values):
        pass

    def _find_single_motor(self, motor, initial_baudrate):
        pass

    def _get_half_turn_homings(self, positions):
        pass

    def _handshake(self):
        pass

    def _split_into_byte_chunks(self, value, length):
        pass

    def broadcast_ping(self, num_retry = 0, raise_on_error = False):
        pass

    def configure_motors(self):
        pass

    def disable_torque(self, motors = None, num_retry = 0):
        if not self.is_connected:
            raise RuntimeError("Cannot disable torque: MotorsBus is not connected.")
        self.api.set_mode(0)  # Set to leader mode
        self.api.set_state(0)  # Set to idle state

    def enable_torque(self, motors = None, num_retry = 0):
        if not self.is_connected:
            raise RuntimeError("Cannot enable torque: MotorsBus is not connected.")
        self.api.set_mode(1)  # Set to follower mode
        self.api.set_state(0)
        # # Enable torque for all motors
        # for motor in self.motor_names:
        #     self.api.set_servo_torque_enable(motor, True)
    
    @property
    def is_calibrated(self) -> bool:
        return self.calibration == self.read_calibration()
    
    def read_calibration(self) -> dict[str, MotorCalibration]:
        offsets = self.sync_read("Homing_Offset", normalize=False)
        mins = self.sync_read("Min_Position_Limit", normalize=False)
        maxes = self.sync_read("Max_Position_Limit", normalize=False)
        drive_modes = self.sync_read("Drive_Mode", normalize=False)

        calibration = {}
        for motor, m in self.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=drive_modes[motor],
                homing_offset=offsets[motor],
                range_min=mins[motor],
                range_max=maxes[motor],
            )

        return calibration
    
    def write_calibration(self, calibration_dict: dict[str, MotorCalibration]) -> None:
        for motor, calibration in calibration_dict.items():
            self.write("Homing_Offset", motor, calibration.homing_offset)
            self.write("Min_Position_Limit", motor, calibration.range_min)
            self.write("Max_Position_Limit", motor, calibration.range_max)

        self.calibration = calibration_dict
    