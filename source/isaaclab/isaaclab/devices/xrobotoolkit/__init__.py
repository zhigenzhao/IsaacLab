# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit-based devices for teleoperation."""

from .retargeters import (
    GMROutputFormat,
    XRGMRRetargeter,
    XRGMRRetargeterCfg,
    XRGripperRetargeter,
    XRGripperRetargeterCfg,
    XRSe3AbsRetargeter,
    XRSe3AbsRetargeterCfg,
    XRSe3RelRetargeter,
    XRSe3RelRetargeterCfg,
    XRT1GMRRetargeter,
    XRT1GMRRetargeterCfg,
    XRT1MinkIKRetargeter,
    XRT1MinkIKRetargeterCfg,
)
from .xr_controller import XRControllerDevice, XRControllerDeviceCfg
from .xr_controller_full_body import XRControllerFullBodyDevice, XRControllerFullBodyDeviceCfg

__all__ = [
    "XRControllerDevice",
    "XRControllerDeviceCfg",
    "XRControllerFullBodyDevice",
    "XRControllerFullBodyDeviceCfg",
    "GMROutputFormat",
    "XRGMRRetargeter",
    "XRGMRRetargeterCfg",
    "XRGripperRetargeter",
    "XRGripperRetargeterCfg",
    "XRSe3AbsRetargeter",
    "XRSe3AbsRetargeterCfg",
    "XRSe3RelRetargeter",
    "XRSe3RelRetargeterCfg",
    "XRT1GMRRetargeter",
    "XRT1GMRRetargeterCfg",
    "XRT1MinkIKRetargeter",
    "XRT1MinkIKRetargeterCfg",
]