# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit controller retargeter using GMR (General Motion Retargeting) for full-body control."""

import numpy as np
import torch
from dataclasses import dataclass
from enum import Enum
from scipy.spatial.transform import Rotation as R
from typing import Any

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg

# Import GMR library
try:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.xrt import estimate_human_height
    GMR_AVAILABLE = True
except ImportError:
    GMR_AVAILABLE = False
    print("Warning: GMR library not available. XRGMRRetargeter will not function.")


# XRoboToolkit joint names (24 joints for full-body tracking)
XRT_JOINT_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1",
    "Left_Knee", "Right_Knee", "Spine2", "Left_Ankle",
    "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head",
    "Left_Shoulder", "Right_Shoulder", "Left_Elbow", "Right_Elbow",
    "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand"
]


class GMROutputFormat(Enum):
    """Output format options for GMR retargeter.

    Determines what data is returned from the retargeter:
    - FULL_QPOS: Complete robot state including root pose and all joints
    - JOINT_POSITIONS_ONLY: Only joint angles, excluding root pose
    """

    FULL_QPOS = "full_qpos"
    """Complete robot state: [root_pos(3), root_quat(4), joint_angles(...)]"""

    JOINT_POSITIONS_ONLY = "joint_positions_only"
    """Only joint angles, excluding root pose: [joint_angles(...)]"""


def quat_mul_np(x: np.ndarray, y: np.ndarray, scalar_first: bool = True) -> np.ndarray:
    """Multiply two quaternions.

    Args:
        x: First quaternion(s) of shape (..., 4)
        y: Second quaternion(s) of shape (..., 4)
        scalar_first: If True, quaternions are in [w, x, y, z] format.
                     If False, quaternions are in [x, y, z, w] format.

    Returns:
        Quaternion multiplication result in same format as input
    """
    if not scalar_first:
        x = x[..., [3, 0, 1, 2]]
        y = y[..., [3, 0, 1, 2]]

    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        x0 * y0 - x1 * y1 - x2 * y2 - x3 * y3,
        x0 * y1 + x1 * y0 + x2 * y3 - x3 * y2,
        x0 * y2 - x1 * y3 + x2 * y0 + x3 * y1,
        x0 * y3 + x1 * y2 - x2 * y1 + x3 * y0
    ], axis=-1)

    if not scalar_first:
        res = res[..., [1, 2, 3, 0]]

    return res


@dataclass
class XRGMRRetargeterCfg(RetargeterCfg):
    """Configuration for XRoboToolkit GMR retargeter.

    This retargeter uses the GMR (General Motion Retargeting) library to perform
    full-body inverse kinematics, mapping 24 body joints from XRoboToolkit tracking
    to robot joint positions.
    """

    robot_type: str = "unitree_g1"
    """Target robot type. Supported: unitree_g1, unitree_h1, booster_t1_29dof, fourier_n1, etc."""

    human_height: float | None = None
    """Human height in meters. If None, auto-estimate from first frame."""

    use_ground_alignment: bool = True
    """If True, automatically align body to ground plane (important for headset-relative tracking)."""

    ground_offset: float = 0.0
    """Manual ground offset adjustment in meters."""

    output_format: GMROutputFormat = GMROutputFormat.FULL_QPOS
    """Output format: FULL_QPOS or JOINT_POSITIONS_ONLY."""

    headless: bool = True
    """If True, run without MuJoCo viewer visualization. If False, display retargeting result in viewer."""

    show_human_skeleton: bool = True
    """If True (and headless=False), display human skeleton visualization alongside robot."""

    viewer_fps: int = 30
    """Frames per second for MuJoCo viewer update (when headless=False)."""


class XRGMRRetargeter(RetargeterBase):
    """Retargets XR full-body tracking to robot using GMR (General Motion Retargeting).

    This retargeter uses the GMR library to perform full-body inverse kinematics,
    mapping 24 body joints from XRoboToolkit tracking to robot joint positions.

    Features:
    - Full-body IK using Mink solver (via GMR)
    - Automatic human height estimation and scaling
    - Ground plane alignment for headset-relative tracking
    - Support for multiple robot types (G1, H1, T1, etc.)
    - Configurable output format with joint selection

    The retargeter handles coordinate transformations from XRoboToolkit's headset frame
    to GMR's world frame, then performs IK to compute robot joint angles.
    """

    def __init__(self, cfg: XRGMRRetargeterCfg):
        """Initialize the GMR retargeter.

        Args:
            cfg: Configuration for the retargeter

        Raises:
            RuntimeError: If GMR library is not available
            ValueError: If selected_joint_names contains invalid joint names (when validated)
        """
        super().__init__(cfg)

        if not GMR_AVAILABLE:
            raise RuntimeError("GMR library not available. Install from /path/to/GMR repository.")

        # Store configuration
        self._robot_type = cfg.robot_type
        self._use_ground_alignment = cfg.use_ground_alignment
        self._ground_offset = cfg.ground_offset
        self._output_format = cfg.output_format
        self._human_height = cfg.human_height
        self._headless = cfg.headless
        self._show_human_skeleton = cfg.show_human_skeleton
        self._viewer_fps = cfg.viewer_fps

        # Lazy initialization (wait for first frame to estimate height if needed)
        self._gmr = None
        self._initialized = False

        # Viewer state (initialized after GMR)
        self._viewer = None

        # Cache for last valid output
        self._last_valid_output = None

        print(f"[XRGMRRetargeter] Initialized for robot: {self._robot_type}")
        print(f"[XRGMRRetargeter] Output format: {self._output_format.value}")
        print(f"[XRGMRRetargeter] Ground alignment: {self._use_ground_alignment}")
        print(f"[XRGMRRetargeter] Headless mode: {self._headless}")
        if self._human_height is not None:
            print(f"[XRGMRRetargeter] Using specified human height: {self._human_height:.2f}m")

    def __del__(self):
        """Destructor to clean up viewer resources."""
        if self._viewer is not None:
            try:
                self._viewer.close()
                print("[XRGMRRetargeter] Viewer closed")
            except Exception:
                pass

    def _initialize_gmr(self, frame_data: dict):
        """Initialize GMR with estimated/configured human height.

        Args:
            frame_data: First frame in GMR format {joint_name: (pos, quat)}

        Raises:
            ValueError: If selected joint names are invalid
        """
        # Estimate height if not specified
        if self._human_height is None:
            self._human_height = estimate_human_height(frame_data)
            print(f"[XRGMRRetargeter] Estimated human height: {self._human_height:.2f}m")

        # Initialize GMR
        self._gmr = GMR(
            src_human="xrt",
            tgt_robot=self._robot_type,
            actual_human_height=self._human_height,
            solver="daqp",
            damping=5e-1,
            verbose=False,
            use_velocity_limit=False
        )

        # Set ground offset if specified
        if self._ground_offset != 0.0:
            self._gmr.set_ground_offset(self._ground_offset)

        # Initialize MuJoCo viewer if not headless
        if not self._headless:
            try:
                from general_motion_retargeting import RobotMotionViewer

                self._viewer = RobotMotionViewer(
                    robot_type=self._robot_type,
                    camera_follow=True,
                    motion_fps=self._viewer_fps,
                    transparent_robot=0,  # 0 = opaque, 1 = transparent
                    record_video=False,
                )
                print("[XRGMRRetargeter] MuJoCo viewer initialized")
            except Exception as e:
                print(f"[XRGMRRetargeter] Warning: Failed to initialize viewer: {e}")
                print("[XRGMRRetargeter] Continuing in headless mode")
                self._viewer = None

        self._initialized = True
        print("[XRGMRRetargeter] GMR initialized successfully")

    def _transform_to_gmr_format(self, body_joints: dict) -> dict:
        """Transform XRControllerFullBodyDevice data to GMR format.

        This method performs coordinate transformation from XRoboToolkit's headset frame
        to GMR's world frame, including both position and orientation transformations.

        Args:
            body_joints: Dictionary from XRControllerFullBodyDevice
                {joint_name: np.array([x, y, z, qx, qy, qz, qw])}
                Positions and orientations in headset frame

        Returns:
            Dictionary in GMR format: {joint_name: (position_np, quaternion_np)}
                Positions and orientations in GMR world frame
                Quaternions in scalar-first format [qw, qx, qy, qz]
        """
        # GMR coordinate transformation matrix (from GMR's xrt.py)
        R_GMR = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])

        frame_data = {}

        for joint_name in XRT_JOINT_NAMES:
            if joint_name not in body_joints:
                continue

            joint_data = body_joints[joint_name]

            # Extract position and quaternion from XR format: [x, y, z, qx, qy, qz, qw]
            position_headset = joint_data[:3]
            quat_xr = joint_data[3:]  # [qx, qy, qz, qw]

            # Convert quaternion to scalar-first format [qw, qx, qy, qz]
            quaternion_headset = np.array([quat_xr[3], quat_xr[0], quat_xr[1], quat_xr[2]])

            # Transform position from headset to GMR world frame
            position_world = position_headset @ R_GMR.T

            # Transform orientation from headset to GMR world frame
            # Rotation: quat_world = R_quat * quat_headset * R_quat_conjugate
            R_quat_scipy = R.from_matrix(R_GMR).as_quat(scalar_first=True)
            quaternion_world = quat_mul_np(R_quat_scipy, quaternion_headset, scalar_first=True)

            # Normalize quaternion
            quaternion_world = quaternion_world / np.linalg.norm(quaternion_world)

            frame_data[joint_name] = (position_world, quaternion_world)

        return frame_data

    def reset(self, **kwargs):
        """Reset retargeter state.

        Called on environment reset. Clears initialization flag to allow
        re-estimation of human height on next frame if height was auto-estimated.

        Args:
            **kwargs: Optional reset parameters (unused by GMR, included for API compatibility)
        """
        # Keep GMR instance but reset initialization flag
        # This allows re-estimation of height if it was auto-estimated
        if self._human_height is None:
            # Height was auto-estimated, allow re-estimation
            self._initialized = False
            print("[XRGMRRetargeter] Reset - will re-estimate height on next frame")
        else:
            # Height was manually specified, keep using it
            print("[XRGMRRetargeter] Reset - keeping manual height setting")

    def retarget(self, data: dict[str, Any]) -> torch.Tensor | None:
        """Convert XR full-body tracking data to robot joint positions.

        Args:
            data: Dictionary from XRControllerFullBodyDevice containing:
                - 'body_joints': {joint_name: [x, y, z, qx, qy, qz, qw]} (24 joints)
                - 'buttons': button states
                - 'timestamp': timestamp
                - 'config': device config

        Returns:
            torch.Tensor | None: Robot state based on output_format, or None if tracking data not ready:
                - FULL_QPOS: [root_pos(3), root_quat(4), joint_angles(...)]
                - JOINT_POSITIONS_ONLY: [joint_angles(...)]
                - None: Tracking data not available yet
        """
        try:
            # Extract body joints
            body_joints = data.get("body_joints", {})

            # Validate data - body_joints is either empty {} or has all 24 joints
            if not body_joints:
                if self._last_valid_output is None:
                    # First call before tracking data available
                    print("[XRGMRRetargeter] Waiting for body tracking data...")
                    return None
                else:
                    # Return cached last valid output
                    return self._last_valid_output

            # Transform to GMR format
            frame_data = self._transform_to_gmr_format(body_joints)

            # Initialize GMR on first frame
            if not self._initialized:
                self._initialize_gmr(frame_data)

            # Run GMR retargeting
            qpos = self._gmr.retarget(frame_data, offset_to_ground=self._use_ground_alignment)

            # Update MuJoCo viewer if active
            if self._viewer is not None:
                try:
                    # Sync viewer with retargeted robot state
                    self._viewer.step(
                        root_pos=qpos[:3],
                        root_rot=qpos[3:7],
                        dof_pos=qpos[7:],
                        human_motion_data=self._gmr.scaled_human_data if self._show_human_skeleton else None,
                        human_pos_offset=np.array([0.0, 0.0, 0.0]),
                        show_human_body_name=False,
                        rate_limit=True,  # Use viewer's internal rate limiter
                    )
                except Exception as e:
                    print(f"[XRGMRRetargeter] Warning: Viewer update failed: {e}")

            # Format output based on configuration
            if self._output_format == GMROutputFormat.FULL_QPOS:
                # Return complete qpos: [root_pos(3), root_quat(4), joint_angles(...)]
                output = qpos

            elif self._output_format == GMROutputFormat.JOINT_POSITIONS_ONLY:
                # Return only joint angles (exclude root pose)
                output = qpos[7:]  # Skip root_pos(3) + root_quat(4)

            else:
                raise ValueError(f"Invalid output_format: {self._output_format}")

            # Convert to torch tensor
            output_tensor = torch.tensor(output, dtype=torch.float32, device=self._sim_device)

            # Cache this valid output for future use
            self._last_valid_output = output_tensor

            return output_tensor

        except Exception as e:
            print(f"[XRGMRRetargeter] Error during retargeting: {e}")
            import traceback
            traceback.print_exc()
            # Return cached output if available, otherwise None
            if self._last_valid_output is not None:
                print("[XRGMRRetargeter] Returning cached output due to error")
                return self._last_valid_output
            else:
                print("[XRGMRRetargeter] No cached output available")
                return None
