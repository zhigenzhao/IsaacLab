# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory to create teleoperation devices from configuration."""

import contextlib
import inspect
import logging
from collections.abc import Callable
from typing import cast

from isaaclab.devices import DeviceBase, DeviceCfg
from isaaclab.devices.retargeter_base import RetargeterBase

# import logger
logger = logging.getLogger(__name__)
from isaaclab.devices.gamepad import Se2Gamepad, Se2GamepadCfg, Se3Gamepad, Se3GamepadCfg
from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg, Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices.openxr.retargeters import (
    G1LowerBodyStandingRetargeter,
    G1LowerBodyStandingRetargeterCfg,
    G1TriHandUpperBodyRetargeter,
    G1TriHandUpperBodyRetargeterCfg,
    GR1T2Retargeter,
    GR1T2RetargeterCfg,
    GripperRetargeter,
    GripperRetargeterCfg,
    Se3AbsRetargeter,
    Se3AbsRetargeterCfg,
    Se3RelRetargeter,
    Se3RelRetargeterCfg,
    UnitreeG1Retargeter,
    UnitreeG1RetargeterCfg,
)
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.devices.spacemouse import Se2SpaceMouse, Se2SpaceMouseCfg, Se3SpaceMouse, Se3SpaceMouseCfg

with contextlib.suppress(ModuleNotFoundError):
    # May fail if xrobotoolkit_sdk is not available
    from isaaclab.devices.xrobotoolkit import (
        XRControllerDevice,
        XRControllerDeviceCfg,
        XRControllerFullBodyDevice,
        XRControllerFullBodyDeviceCfg,
        XRG1MinkIKRetargeter,
        XRG1MinkIKRetargeterCfg,
        XRGripperRetargeter,
        XRGripperRetargeterCfg,
        XRInspireHandRetargeter,
        XRInspireHandRetargeterCfg,
        XRSe3AbsRetargeter,
        XRSe3AbsRetargeterCfg,
        XRSe3RelRetargeter,
        XRSe3RelRetargeterCfg,
        XRT1GMRRetargeter,
        XRT1GMRRetargeterCfg,
        XRT1MinkIKRetargeter,
        XRT1MinkIKRetargeterCfg,
        XRTwistRetargeter,
        XRTwistRetargeterCfg,
    )

with contextlib.suppress(ModuleNotFoundError):
    # May fail if xr is not in use
    from isaaclab.devices.openxr import ManusVive, ManusViveCfg, OpenXRDevice, OpenXRDeviceCfg

# Map device types to their constructor and expected config type
DEVICE_MAP: dict[type[DeviceCfg], type[DeviceBase]] = {
    Se3KeyboardCfg: Se3Keyboard,
    Se3SpaceMouseCfg: Se3SpaceMouse,
    Se3GamepadCfg: Se3Gamepad,
    Se2KeyboardCfg: Se2Keyboard,
    Se2GamepadCfg: Se2Gamepad,
    Se2SpaceMouseCfg: Se2SpaceMouse,
    OpenXRDeviceCfg: OpenXRDevice,
    ManusViveCfg: ManusVive,
}

# Add XRoboToolkit devices if available
with contextlib.suppress(NameError):
    DEVICE_MAP[XRControllerDeviceCfg] = XRControllerDevice
    DEVICE_MAP[XRControllerFullBodyDeviceCfg] = XRControllerFullBodyDevice


# Map configuration types to their corresponding retargeter classes
RETARGETER_MAP: dict[type[RetargeterCfg], type[RetargeterBase]] = {
    Se3AbsRetargeterCfg: Se3AbsRetargeter,
    Se3RelRetargeterCfg: Se3RelRetargeter,
    GripperRetargeterCfg: GripperRetargeter,
    GR1T2RetargeterCfg: GR1T2Retargeter,
    G1TriHandUpperBodyRetargeterCfg: G1TriHandUpperBodyRetargeter,
    G1LowerBodyStandingRetargeterCfg: G1LowerBodyStandingRetargeter,
    UnitreeG1RetargeterCfg: UnitreeG1Retargeter,
}

# Add XRoboToolkit retargeters if available
with contextlib.suppress(NameError):
    RETARGETER_MAP[XRSe3RelRetargeterCfg] = XRSe3RelRetargeter
    RETARGETER_MAP[XRSe3AbsRetargeterCfg] = XRSe3AbsRetargeter
    RETARGETER_MAP[XRG1MinkIKRetargeterCfg] = XRG1MinkIKRetargeter
    RETARGETER_MAP[XRGripperRetargeterCfg] = XRGripperRetargeter
    RETARGETER_MAP[XRInspireHandRetargeterCfg] = XRInspireHandRetargeter
    RETARGETER_MAP[XRT1GMRRetargeterCfg] = XRT1GMRRetargeter
    RETARGETER_MAP[XRT1MinkIKRetargeterCfg] = XRT1MinkIKRetargeter
    RETARGETER_MAP[XRTwistRetargeterCfg] = XRTwistRetargeter


def create_teleop_device(
    device_name: str, devices_cfg: dict[str, DeviceCfg], callbacks: dict[str, Callable] | None = None
) -> DeviceBase:
    """Create a teleoperation device based on configuration.

    Args:
        device_name: The name of the device to create (must exist in devices_cfg)
        devices_cfg: Dictionary of device configurations
        callbacks: Optional dictionary of callbacks to register with the device
            Keys are the button/gesture names, values are callback functions

    Returns:
        The configured teleoperation device

    Raises:
        ValueError: If the device name is not found in the configuration
        ValueError: If the device configuration type is not supported
    """
    if device_name not in devices_cfg:
        raise ValueError(f"Device '{device_name}' not found in teleop device configurations")

    device_cfg = devices_cfg[device_name]
    callbacks = callbacks or {}

    # Determine constructor from the configuration itself
    device_constructor = getattr(device_cfg, "class_type", None)
    if device_constructor is None:
        raise ValueError(
            f"Device configuration '{device_name}' does not declare class_type. "
            "Set cfg.class_type to the concrete DeviceBase subclass."
        )
    if not issubclass(device_constructor, DeviceBase):
        raise TypeError(f"class_type for '{device_name}' must be a subclass of DeviceBase; got {device_constructor}")

    # Try to create retargeters if they are configured
    retargeters = []
    if hasattr(device_cfg, "retargeters") and device_cfg.retargeters is not None:
        try:
            # Create retargeters based on configuration using per-config retargeter_type
            for retargeter_cfg in device_cfg.retargeters:
                retargeter_constructor = getattr(retargeter_cfg, "retargeter_type", None)
                if retargeter_constructor is None:
                    raise ValueError(
                        f"Retargeter configuration {type(retargeter_cfg).__name__} does not declare retargeter_type. "
                        "Set cfg.retargeter_type to the concrete RetargeterBase subclass."
                    )
                if not issubclass(retargeter_constructor, RetargeterBase):
                    raise TypeError(
                        f"retargeter_type for {type(retargeter_cfg).__name__} must be a subclass of RetargeterBase; got"
                        f" {retargeter_constructor}"
                    )
                retargeters.append(retargeter_constructor(retargeter_cfg))

        except NameError as e:
            raise ValueError(f"Failed to create retargeters: {e}")

    # Build constructor kwargs based on signature
    constructor_params = inspect.signature(device_constructor).parameters
    params: dict = {"cfg": device_cfg}
    if "retargeters" in constructor_params:
        params["retargeters"] = retargeters
    device = cast(DeviceBase, device_constructor(**params))

    # Register callbacks
    for key, callback in callbacks.items():
        device.add_callback(key, callback)

    logging.info(f"Created teleoperation device: {device_name}")
    return device
