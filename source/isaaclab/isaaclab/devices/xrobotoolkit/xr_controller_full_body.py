# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit XR controller for full-body motion capture."""

import numpy as np
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..device_base import DeviceBase, DeviceCfg

# Import XRoboToolkit SDK
try:
    import xrobotoolkit_sdk as xrt
    XRT_AVAILABLE = True
except ImportError:
    XRT_AVAILABLE = False
    print("Warning: xrobotoolkit_sdk not available. XRControllerFullBodyDevice will not function properly.")


# XRoboToolkit joint names (24 joints for full-body tracking)
XRT_JOINT_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1",
    "Left_Knee", "Right_Knee", "Spine2", "Left_Ankle",
    "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head",
    "Left_Shoulder", "Right_Shoulder", "Left_Elbow", "Right_Elbow",
    "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand"
]


class XRControllerFullBodyDevice(DeviceBase):
    """XRoboToolkit full-body tracking device for recording body joint states and button inputs.

    This class implements a full-body motion capture interface using the XRoboToolkit SDK
    to capture 24 body joint poses and button states. It provides raw data in the headset
    coordinate frame without any transformations, suitable for recording and future retargeting.

    Raw data format (_get_raw_data output):
    * A dictionary containing body joint poses and button states
    * Dictionary keys are XRControllerFullBodyDeviceValues enum members
    * Body joints as dictionary mapping joint names to 7-element arrays: [x, y, z, qx, qy, qz, qw]
    * Button states as boolean values
    * All data in headset coordinate frame (no transformations applied)

    Body joint names (24 joints):
    * Lower body: Pelvis, Left/Right Hip, Left/Right Knee, Left/Right Ankle, Left/Right Foot
    * Spine: Spine1, Spine2, Spine3
    * Upper body: Neck, Head, Left/Right Collar, Left/Right Shoulder
    * Arms: Left/Right Elbow, Left/Right Wrist, Left/Right Hand
    """

    class XRControllerFullBodyDeviceValues(Enum):
        """Enum for XR full-body device data keys.

        Provides type-safe keys for accessing device data in the dictionary returned
        by _get_raw_data(). This enables IDE autocomplete and prevents typos.
        """
        BODY_JOINTS = "body_joints"              # Dict {joint_name: [x, y, z, qx, qy, qz, qw]}
        BUTTONS = "buttons"                      # Dictionary of button states
        LEFT_TRIGGER = "left_trigger"            # Left trigger value [0-1]
        RIGHT_TRIGGER = "right_trigger"          # Right trigger value [0-1]
        LEFT_GRIP = "left_grip"                  # Left grip value [0-1]
        RIGHT_GRIP = "right_grip"                # Right grip value [0-1]
        TIMESTAMP = "timestamp"                  # Timestamp in nanoseconds
        CONFIG = "config"                        # Device configuration dictionary

    def __init__(self, cfg: "XRControllerFullBodyDeviceCfg", retargeters: list | None = None):
        """Initialize the XR full-body tracking device.

        Args:
            cfg: Configuration object for XR full-body tracking settings.
            retargeters: List of retargeter instances to transform raw data into robot commands.
        """
        super().__init__(retargeters)

        if not XRT_AVAILABLE:
            raise RuntimeError("xrobotoolkit_sdk is not available. Cannot initialize XRControllerFullBodyDevice.")

        # Store configuration
        self.cfg = cfg
        self._sim_device = cfg.sim_device

        # Initialize XRoboToolkit SDK
        try:
            xrt.init()
            print("XRoboToolkit SDK initialized successfully for full-body tracking.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize XRoboToolkit SDK: {e}")

        # Internal state for button tracking
        self._button_pressed_prev = {}

        # Callback dictionary
        self._additional_callbacks = dict()

        print("XR Full-Body Controller initialized")

    def __del__(self):
        """Destructor for the class."""
        if XRT_AVAILABLE:
            try:
                xrt.close()
                print("XRoboToolkit SDK closed.")
            except Exception:
                pass

    def __str__(self) -> str:
        """Returns: A string containing the information of the XR full-body controller."""
        msg = f"XRoboToolkit XR Full-Body Controller: {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tBody Tracking: 24 joints in headset frame\n"
        msg += "\tJoints: Pelvis, Hips, Knees, Ankles, Feet, Spine,\n"
        msg += "\t        Neck, Head, Collars, Shoulders, Elbows, Wrists, Hands\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tButton Mappings for Demo Recording:\n"
        msg += "\t\tA button: START recording\n"
        msg += "\t\tB button: SAVE and reset (saves current episode)\n"
        msg += "\t\tX button: RESET environment (discards data)\n"
        msg += "\t\tY button: PAUSE recording\n"
        msg += "\t\tRight joystick click: DISCARD recording\n"
        return msg

    def reset(self):
        """Reset the internal state."""
        self._button_pressed_prev = {}

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind to controller buttons.

        Args:
            key: The button to bind to. Supported: 'START', 'SAVE', 'RESET', 'PAUSE', 'DISCARD'.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def _get_raw_data(self) -> dict[str, Any]:
        """Get raw full-body tracking data from XRoboToolkit.

        Returns:
            Dictionary containing:
                - body_joints: dict {joint_name: [x, y, z, qx, qy, qz, qw]} (24 joints in headset frame)
                - buttons: dict with button states
                - left_trigger: float [0-1]
                - right_trigger: float [0-1]
                - left_grip: float [0-1]
                - right_grip: float [0-1]
                - timestamp: int (nanoseconds)
                - config: dict with device configuration
        """
        try:
            # Get body joint data
            body_joints = {}
            if xrt.is_body_data_available():
                # Get body tracking data from XRoboToolkit
                # Returns list of 24 joint poses, each as [x, y, z, qx, qy, qz, qw]
                body_poses = xrt.get_body_joints_pose()

                if body_poses is not None and len(body_poses) == 24:
                    for i, joint_name in enumerate(XRT_JOINT_NAMES):
                        joint_data = body_poses[i]
                        if len(joint_data) >= 7:
                            # Store as [x, y, z, qx, qy, qz, qw] in headset frame (no transformation)
                            body_joints[joint_name] = np.array(joint_data[:7], dtype=np.float32)

            # Get input states (triggers and grips)
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

            # Build complete data dictionary
            data = {
                self.XRControllerFullBodyDeviceValues.BODY_JOINTS.value: body_joints,
                self.XRControllerFullBodyDeviceValues.BUTTONS.value: buttons,
                self.XRControllerFullBodyDeviceValues.LEFT_TRIGGER.value: left_trigger,
                self.XRControllerFullBodyDeviceValues.RIGHT_TRIGGER.value: right_trigger,
                self.XRControllerFullBodyDeviceValues.LEFT_GRIP.value: left_grip,
                self.XRControllerFullBodyDeviceValues.RIGHT_GRIP.value: right_grip,
                self.XRControllerFullBodyDeviceValues.TIMESTAMP.value: xrt.get_time_stamp_ns(),
                self.XRControllerFullBodyDeviceValues.CONFIG.value: {},
            }

            return data

        except Exception as e:
            print(f"Error getting XR full-body tracking data: {e}")
            # Return default values on error
            return {
                self.XRControllerFullBodyDeviceValues.BODY_JOINTS.value: {},
                self.XRControllerFullBodyDeviceValues.BUTTONS.value: {
                    k: False for k in ['left_primary', 'right_primary', 'left_secondary',
                                      'right_secondary', 'left_menu', 'right_menu',
                                      'left_axis_click', 'right_axis_click']
                },
                self.XRControllerFullBodyDeviceValues.LEFT_TRIGGER.value: 0.0,
                self.XRControllerFullBodyDeviceValues.RIGHT_TRIGGER.value: 0.0,
                self.XRControllerFullBodyDeviceValues.LEFT_GRIP.value: 0.0,
                self.XRControllerFullBodyDeviceValues.RIGHT_GRIP.value: 0.0,
                self.XRControllerFullBodyDeviceValues.TIMESTAMP.value: 0,
                self.XRControllerFullBodyDeviceValues.CONFIG.value: {},
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
                # A button = START recording
                if button_name == 'right_primary' and 'START' in self._additional_callbacks:
                    self._additional_callbacks['START']()
                # B button = SAVE and reset
                elif button_name == 'right_secondary' and 'SAVE' in self._additional_callbacks:
                    self._additional_callbacks['SAVE']()
                # X button = RESET
                elif button_name == 'left_primary' and 'RESET' in self._additional_callbacks:
                    self._additional_callbacks['RESET']()
                # Y button = PAUSE recording
                elif button_name == 'left_secondary' and 'PAUSE' in self._additional_callbacks:
                    self._additional_callbacks['PAUSE']()
                # Right joystick click = DISCARD
                elif button_name == 'right_axis_click' and 'DISCARD' in self._additional_callbacks:
                    self._additional_callbacks['DISCARD']()

        # Update previous button states
        self._button_pressed_prev = buttons.copy()


@dataclass
class XRControllerFullBodyDeviceCfg(DeviceCfg):
    """Configuration for XRoboToolkit full-body tracking device."""

    class_type: type[DeviceBase] = XRControllerFullBodyDevice
