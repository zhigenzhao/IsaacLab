# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T1 Upper Body GMR retargeter - wrapper around XRGMRRetargeter for T1 upper body control."""

import torch
from dataclasses import dataclass
from typing import Any

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from .xr_gmr_retargeter import XRGMRRetargeter, XRGMRRetargeterCfg, GMROutputFormat


@dataclass
class XRT1GMRRetargeterCfg(RetargeterCfg):
    """Configuration for T1 upper body GMR retargeter.

    This retargeter wraps XRGMRRetargeter to provide T1-specific upper body control.
    It performs full-body GMR retargeting and extracts only the 16 upper body joints
    (2 head + 14 arms) to match XRT1MinkIKRetargeter output format.

    Note: Head joints default to [0.0, 0.0] as GMR's T1 model doesn't include head tracking.
    """

    human_height: float | None = None
    """Human height in meters. If None, auto-estimate from first frame."""

    use_ground_alignment: bool = True
    """If True, automatically align body to ground plane (important for headset-relative tracking)."""

    ground_offset: float = 0.0
    """Manual ground offset adjustment in meters."""

    headless: bool = True
    """If True, run without MuJoCo viewer visualization. If False, display retargeting result in viewer."""

    show_human_skeleton: bool = True
    """If True (and headless=False), display human skeleton visualization alongside robot."""

    viewer_fps: int = 30
    """Frames per second for MuJoCo viewer update (when headless=False)."""

    use_threading: bool = True
    """If True, run GMR retargeting in a background thread for non-blocking operation."""

    thread_rate_hz: float = 90.0
    """Update rate for background GMR thread in Hz (when use_threading=True)."""


class XRT1GMRRetargeter(RetargeterBase):
    """T1 Upper Body GMR Retargeter - Wrapper for full-body GMR with upper body extraction.

    This retargeter wraps XRGMRRetargeter to provide T1-specific upper body joint control.
    It performs full-body GMR retargeting for the T1 humanoid robot and extracts only
    the 16 upper body joints to match XRT1MinkIKRetargeter output format.

    Architecture:
    - Uses XRGMRRetargeter internally for full-body IK
    - Extracts 14 arm joints from GMR output (qpos[7:21])
    - Adds 2 default head joints [0.0, 0.0] at the beginning
    - Returns 16-element tensor matching XRT1MinkIKRetargeter format

    Output format (16 joints):
    - [0:2]: Head joints (AAHead_yaw, Head_pitch) - defaults to [0.0, 0.0]
    - [2:9]: Left arm (7 joints: Shoulder_Pitch/Roll, Elbow_Pitch/Yaw, Wrist_Pitch/Yaw, Hand_Roll)
    - [9:16]: Right arm (7 joints: same as left arm)

    Note: GMR's T1 29DoF model does not include head joints in the XML, so head
    joints are set to zero. For head tracking, use XRT1MinkIKRetargeter instead.
    """

    def __init__(self, cfg: XRT1GMRRetargeterCfg):
        """Initialize the T1 GMR retargeter wrapper.

        Args:
            cfg: Configuration for the retargeter
        """
        super().__init__(cfg)

        # Create internal GMR retargeter configured for T1 robot
        gmr_cfg = XRGMRRetargeterCfg(
            robot_type="booster_t1_29dof",
            output_format=GMROutputFormat.FULL_QPOS,  # Need full qpos to extract arm joints
            human_height=cfg.human_height,
            use_ground_alignment=cfg.use_ground_alignment,
            ground_offset=cfg.ground_offset,
            headless=cfg.headless,
            show_human_skeleton=cfg.show_human_skeleton,
            viewer_fps=cfg.viewer_fps,
            use_threading=cfg.use_threading,
            thread_rate_hz=cfg.thread_rate_hz,
            sim_device=cfg.sim_device,
        )
        self._gmr_retargeter = XRGMRRetargeter(gmr_cfg)

        print("[XRT1GMRRetargeter] Initialized T1 upper body GMR retargeter")
        print("[XRT1GMRRetargeter] Output: 16 joints (2 head + 14 arms)")
        print("[XRT1GMRRetargeter] Note: Head joints default to [0.0, 0.0] (no head tracking)")

    def __del__(self):
        """Destructor to clean up internal retargeter."""
        # Internal retargeter will clean up itself
        pass

    def reset(self, **kwargs):
        """Reset retargeter state (called on environment reset).

        Args:
            **kwargs: Optional reset parameters (passed through to internal GMR retargeter)
        """
        if self._gmr_retargeter is not None:
            self._gmr_retargeter.reset(**kwargs)
        print("[XRT1GMRRetargeter] Reset")

    def retarget(self, data: dict[str, Any]) -> torch.Tensor | None:
        """Convert XR full-body tracking data to T1 upper body joint positions.

        This method:
        1. Calls internal GMR retargeter for full-body IK
        2. Extracts 14 arm joints from GMR output (qpos[7:21])
        3. Prepends 2 default head joints [0.0, 0.0]
        4. Returns 16-element tensor matching XRT1MinkIKRetargeter format

        Args:
            data: Dictionary from XRControllerFullBodyDevice containing:
                - 'body_joints': {joint_name: [x, y, z, qx, qy, qz, qw]} (24 joints)
                - 'buttons': button states
                - 'left_grip', 'right_grip': grip values
                - 'timestamp': timestamp

        Returns:
            torch.Tensor: 16-element tensor with upper body joint positions
                [0:2]   - Head joints (AAHead_yaw, Head_pitch) = [0.0, 0.0]
                [2:9]   - Left arm (7 joints)
                [9:16]  - Right arm (7 joints)
        """
        try:
            # Get full GMR retargeting output
            # For T1 29DoF: qpos shape is (34,)
            # qpos[0:7]   = Root pose (3 pos + 4 quat)
            # qpos[7:21]  = 14 arm joints (7 left + 7 right) <-- What we want
            # qpos[21]    = Waist
            # qpos[22:34] = 12 leg joints
            gmr_qpos = self._gmr_retargeter.retarget(data)

            # If GMR retargeter returns None (no tracking data yet), pass it through
            if gmr_qpos is None:
                return None

            # Extract 14 arm joints from GMR output (torch tensor slicing)
            # qpos[7:21] contains: [Left_Shoulder_Pitch, Left_Shoulder_Roll, ..., Right_Hand_Roll]
            arm_joints = gmr_qpos[7:21]  # 14 joints

            # Create default head joints (no head tracking in GMR T1 model)
            head_joints = torch.zeros(2, dtype=torch.float32, device=self._sim_device)
            head_joints[1] = 1.0472

            # Combine: [head(2) + arms(14)] = 16 upper body joints
            # This matches XRT1MinkIKRetargeter output format
            upper_body_joints = torch.cat([head_joints, arm_joints])

            return upper_body_joints

        except Exception as e:
            print(f"[XRT1GMRRetargeter] Error during retargeting: {e}")
            import traceback
            traceback.print_exc()
            # Return None - no valid data available
            return None
