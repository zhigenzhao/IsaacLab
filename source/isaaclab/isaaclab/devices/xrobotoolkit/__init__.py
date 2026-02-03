# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit-based devices for teleoperation."""

from .retargeters import (
    GMROutputFormat,
    TwistOutputFormat,
    XRG1MinkIKRetargeter,
    XRG1MinkIKRetargeterCfg,
    XRGMRRetargeter,
    XRGMRRetargeterCfg,
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
    XRTwist2G1Retargeter,
    XRTwist2G1RetargeterCfg,
)
from .xr_controller import XRControllerDevice, XRControllerDeviceCfg
from .xr_controller_full_body import XRControllerFullBodyDevice, XRControllerFullBodyDeviceCfg

__all__ = [
    "XRControllerDevice",
    "XRControllerDeviceCfg",
    "XRControllerFullBodyDevice",
    "XRControllerFullBodyDeviceCfg",
    "GMROutputFormat",
    "TwistOutputFormat",
    "XRG1MinkIKRetargeter",
    "XRG1MinkIKRetargeterCfg",
    "XRGMRRetargeter",
    "XRGMRRetargeterCfg",
    "XRGripperRetargeter",
    "XRGripperRetargeterCfg",
    "XRInspireHandRetargeter",
    "XRInspireHandRetargeterCfg",
    "XRSe3AbsRetargeter",
    "XRSe3AbsRetargeterCfg",
    "XRSe3RelRetargeter",
    "XRSe3RelRetargeterCfg",
    "XRT1GMRRetargeter",
    "XRT1GMRRetargeterCfg",
    "XRT1MinkIKRetargeter",
    "XRT1MinkIKRetargeterCfg",
    "XRTwistRetargeter",
    "XRTwistRetargeterCfg",
    "XRTwist2G1Retargeter",
    "XRTwist2G1RetargeterCfg",
]