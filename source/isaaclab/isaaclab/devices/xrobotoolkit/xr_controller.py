# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit XR controller for SE(3) control."""

import numpy as np
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..device_base import DeviceBase, DeviceCfg

# Import XRoboToolkit SDK
try:
    import xrobotoolkit_sdk as xrt
    XRT_AVAILABLE = True
except ImportError:
    XRT_AVAILABLE = False
    print("Warning: xrobotoolkit_sdk not available. XRControllerDevice will not function properly.")


@dataclass
class XRControllerDeviceCfg(DeviceCfg):
    """Configuration for XRoboToolkit XR controller devices."""

    pos_sensitivity: float = 0.4
    """Sensitivity for positional control (m/s)."""

    rot_sensitivity: float = 0.8
    """Sensitivity for rotational control (rad/s)."""

    control_mode: str = "right_hand"
    """Control mode: 'right_hand', 'left_hand', or 'dual_hand'."""

    gripper_source: str = "trigger"
    """Gripper control source: 'trigger', 'grip', or 'button'."""

    deadzone_threshold: float = 0.05
    """Minimum movement threshold to filter out noise."""


class XRControllerDevice(DeviceBase):
    """XRoboToolkit XR controller for sending SE(3) commands as delta poses.

    This class implements an XR controller interface using the XRoboToolkit SDK to provide
    commands to a robotic arm with a gripper. It tracks VR/AR controller poses and maps
    them to robot control commands via the _get_raw_data() method and retargeters.

    Raw data format (_get_raw_data output):
    * A dictionary containing controller poses, input states, and button states
    * Controller poses as 7-element arrays: [x, y, z, qw, qx, qy, qz]
    * Input values (triggers, grips) as floats [0-1]
    * Button states as boolean values

    Control modes:
    * right_hand: Uses right controller for pose control
    * left_hand: Uses left controller for pose control
    * dual_hand: Uses both controllers

    Gripper sources:
    * trigger: Uses trigger value for gripper control
    * grip: Uses grip value for gripper control
    * button: Uses primary button for gripper toggle
    """

    def __init__(self, cfg: XRControllerDeviceCfg):
        """Initialize the XR controller device.

        Args:
            cfg: Configuration object for XR controller settings.
        """
        super().__init__()

        if not XRT_AVAILABLE:
            raise RuntimeError("xrobotoolkit_sdk is not available. Cannot initialize XRControllerDevice.")

        # Store configuration
        self.cfg = cfg
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.control_mode = cfg.control_mode
        self.gripper_source = cfg.gripper_source
        self.deadzone_threshold = cfg.deadzone_threshold
        self._sim_device = cfg.sim_device

        # Initialize XRoboToolkit SDK
        try:
            xrt.init()
            print("XRoboToolkit SDK initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize XRoboToolkit SDK: {e}")

        # Internal state for button tracking
        self._button_pressed_prev = {}

        # Callback dictionary
        self._additional_callbacks = dict()

        print(f"XR Controller initialized with mode: {self.control_mode}, gripper source: {self.gripper_source}")

    def __del__(self):
        """Destructor for the class."""
        if XRT_AVAILABLE:
            try:
                xrt.close()
                print("XRoboToolkit SDK closed.")
            except Exception:
                pass

    def __str__(self) -> str:
        """Returns: A string containing the information of the XR controller."""
        msg = f"XRoboToolkit XR Controller: {self.__class__.__name__}\n"
        msg += f"\tControl Mode: {self.control_mode}\n"
        msg += f"\tGripper Source: {self.gripper_source}\n"
        msg += f"\tPosition Sensitivity: {self.pos_sensitivity}\n"
        msg += f"\tRotation Sensitivity: {self.rot_sensitivity}\n"
        msg += f"\tDeadzone Threshold: {self.deadzone_threshold}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tController Usage:\n"
        if self.control_mode == "right_hand":
            msg += "\t\tRight controller: Move to control robot end-effector\n"
        elif self.control_mode == "left_hand":
            msg += "\t\tLeft controller: Move to control robot end-effector\n"
        else:
            msg += "\t\tBoth controllers: Dual-hand control\n"

        if self.gripper_source == "trigger":
            msg += "\t\tTrigger: Squeeze to close gripper\n"
        elif self.gripper_source == "grip":
            msg += "\t\tGrip: Squeeze to close gripper\n"
        else:
            msg += "\t\tPrimary button: Press to toggle gripper\n"

        msg += "\t\tMenu button: Reset pose reference\n"
        return msg

    def reset(self):
        """Reset the internal state."""
        self._button_pressed_prev = {}

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind to controller buttons.

        Args:
            key: The button to bind to. Supported: 'RESET', 'START', 'STOP'.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def _get_raw_data(self) -> dict[str, Any]:
        """Get raw controller data from XRoboToolkit.

        Returns:
            Dictionary containing:
                - left_controller: [x, y, z, qw, qx, qy, qz] pose array
                - right_controller: [x, y, z, qw, qx, qy, qz] pose array
                - left_trigger: float [0-1]
                - right_trigger: float [0-1]
                - left_grip: float [0-1]
                - right_grip: float [0-1]
                - buttons: dict with button states
                - config: dict with device configuration
        """
        try:
            # Get controller poses
            left_pose = np.array(xrt.get_left_controller_pose(), dtype=np.float32)
            right_pose = np.array(xrt.get_right_controller_pose(), dtype=np.float32)

            # Get input states
            left_trigger = xrt.get_left_trigger()
            right_trigger = xrt.get_right_trigger()
            left_grip = xrt.get_left_grip()
            right_grip = xrt.get_right_grip()

            # Get button states
            buttons = {
                'left_primary': xrt.get_X_button(),
                'right_primary': xrt.get_A_button(),
                'left_secondary': xrt.get_Y_button(),
                'right_secondary': xrt.get_B_button(),
                'left_menu': xrt.get_left_menu_button(),
                'right_menu': xrt.get_right_menu_button(),
                'left_axis_click': xrt.get_left_axis_click(),
                'right_axis_click': xrt.get_right_axis_click(),
            }

            # Handle button callbacks
            self._handle_button_callbacks(buttons)

            return {
                'left_controller': left_pose,
                'right_controller': right_pose,
                'left_trigger': left_trigger,
                'right_trigger': right_trigger,
                'left_grip': left_grip,
                'right_grip': right_grip,
                'buttons': buttons,
                'timestamp': xrt.get_time_stamp_ns(),
                'config': {
                    'pos_sensitivity': self.pos_sensitivity,
                    'rot_sensitivity': self.rot_sensitivity,
                    'control_mode': self.control_mode,
                    'gripper_source': self.gripper_source,
                    'deadzone_threshold': self.deadzone_threshold
                }
            }

        except Exception as e:
            print(f"Error getting XR controller data: {e}")
            # Return default values on error
            default_pose = np.zeros(7, dtype=np.float32)
            return {
                'left_controller': default_pose,
                'right_controller': default_pose,
                'left_trigger': 0.0,
                'right_trigger': 0.0,
                'left_grip': 0.0,
                'right_grip': 0.0,
                'buttons': {k: False for k in ['left_primary', 'right_primary', 'left_secondary',
                                               'right_secondary', 'left_menu', 'right_menu',
                                               'left_axis_click', 'right_axis_click']},
                'timestamp': 0,
                'config': {
                    'pos_sensitivity': self.pos_sensitivity,
                    'rot_sensitivity': self.rot_sensitivity,
                    'control_mode': self.control_mode,
                    'gripper_source': self.gripper_source,
                    'deadzone_threshold': self.deadzone_threshold
                }
            }

    def _handle_button_callbacks(self, buttons: dict[str, bool]) -> None:
        """Handle button press callbacks.

        Args:
            buttons: Dictionary of current button states
        """
        # Check for button press events (rising edge)
        for button_name, is_pressed in buttons.items():
            was_pressed = self._button_pressed_prev.get(button_name, False)

            if is_pressed and not was_pressed:  # Rising edge
                if button_name in ['left_menu', 'right_menu'] and 'RESET' in self._additional_callbacks:
                    self._additional_callbacks['RESET']()
                elif button_name in ['left_secondary', 'right_secondary'] and 'START' in self._additional_callbacks:
                    self._additional_callbacks['START']()
                elif button_name in ['left_axis_click', 'right_axis_click'] and 'STOP' in self._additional_callbacks:
                    self._additional_callbacks['STOP']()

        # Update previous button states
        self._button_pressed_prev = buttons.copy()