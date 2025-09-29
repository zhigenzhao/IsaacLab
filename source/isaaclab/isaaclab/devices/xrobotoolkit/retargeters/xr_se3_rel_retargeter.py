# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit controller retargeter for relative SE(3) control."""

import numpy as np
import torch
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from typing import Any

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG


@dataclass
class XRSe3RelRetargeterCfg(RetargeterCfg):
    """Configuration for XRoboToolkit relative position retargeter."""

    control_hand: str = "right"
    """Which hand to use for control: 'left' or 'right'."""

    pos_scale_factor: float = 1.0
    """Amplification factor for position changes (higher = larger robot movements)."""

    rot_scale_factor: float = 1.0
    """Amplification factor for rotation changes (higher = larger robot rotations)."""

    activation_source: str = "grip"
    """Input source for activation: 'grip' or 'trigger'."""

    activation_threshold: float = 0.9
    """Threshold value [0-1] for activation."""

    alpha_pos: float = 0.5
    """Position smoothing parameter (0-1); higher values track more closely."""

    alpha_rot: float = 0.5
    """Rotation smoothing parameter (0-1); higher values track more closely."""

    zero_out_xy_rotation: bool = False
    """If True, ignore rotations around x and y axes, allowing only z-axis rotation."""

    enable_visualization: bool = False
    """If True, show a visual marker representing the target end-effector pose."""

    R_xr_to_world: np.ndarray | None = None
    """Rotation matrix to transform XR frame to world frame. If None, uses identity."""


class XRSe3RelRetargeter(RetargeterBase):
    """Retargets XR controller data to end-effector commands using relative positioning.

    This retargeter calculates delta poses between consecutive controller poses to generate
    incremental robot movements. It activates when the specified button (grip or trigger)
    is pressed beyond the threshold.

    Features:
    - Delta-based control with configurable scaling
    - Optional constraint to zero out X/Y rotations (keeping only Z-axis rotation)
    - Motion smoothing with adjustable parameters
    - Optional visualization of the target end-effector pose
    - Coordinate frame transformation support
    """

    def __init__(self, cfg: XRSe3RelRetargeterCfg):
        """Initialize the relative motion retargeter.

        Args:
            cfg: Configuration for the retargeter
        """
        super().__init__(cfg)

        # Store configuration
        if cfg.control_hand not in ["left", "right"]:
            raise ValueError("control_hand must be 'left' or 'right'")
        self._control_hand = cfg.control_hand
        self._pos_scale_factor = cfg.pos_scale_factor
        self._rot_scale_factor = cfg.rot_scale_factor
        self._activation_source = cfg.activation_source
        self._activation_threshold = cfg.activation_threshold
        self._alpha_pos = cfg.alpha_pos
        self._alpha_rot = cfg.alpha_rot
        self._zero_out_xy_rotation = cfg.zero_out_xy_rotation

        # Set up coordinate transformation
        if cfg.R_xr_to_world is not None:
            self._R_xr_to_world = cfg.R_xr_to_world
        else:
            self._R_xr_to_world = np.eye(3)

        # State tracking
        self._is_active = False
        self._ref_controller_pos = None
        self._ref_controller_quat = None
        self._smoothed_delta_pos = np.zeros(3)
        self._smoothed_delta_rot = np.zeros(3)

        # Define thresholds for small movements
        self._position_threshold = 0.001
        self._rotation_threshold = 0.01

        # Initialize visualization if enabled
        self._enable_visualization = cfg.enable_visualization
        if cfg.enable_visualization:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self._goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/xr_ee_goal"))
            self._goal_marker.set_visibility(True)
            self._visualization_pos = np.zeros(3)
            self._visualization_rot = np.array([1.0, 0.0, 0.0, 0.0])

    def retarget(self, data: dict[str, Any]) -> torch.Tensor:
        """Convert controller data to robot end-effector command.

        Args:
            data: Dictionary containing controller data from XRControllerDevice:
                - 'left_controller': [x, y, z, qx, qy, qz, qw] pose array
                - 'right_controller': [x, y, z, qx, qy, qz, qw] pose array
                - 'left_trigger': float [0-1]
                - 'right_trigger': float [0-1]
                - 'left_grip': float [0-1]
                - 'right_grip': float [0-1]

        Returns:
            torch.Tensor: 6D tensor containing position delta (xyz) and rotation delta (rx,ry,rz)
        """
        # Get controller pose based on control hand
        controller_key = f"{self._control_hand}_controller"
        controller_pose = data.get(controller_key)

        if controller_pose is None:
            # Return zero deltas if no controller data
            return torch.zeros(6, dtype=torch.float32, device=self._sim_device)

        # Extract position and quaternion from controller pose
        # XRoboToolkit format: [x, y, z, qx, qy, qz, qw]
        controller_pos = controller_pose[:3]
        controller_quat = np.array([controller_pose[6], controller_pose[3], controller_pose[4], controller_pose[5]])  # Convert to [qw, qx, qy, qz]

        # Check activation status
        activation_key = f"{self._control_hand}_{self._activation_source}"
        activation_value = data.get(activation_key, 0.0)
        is_active = activation_value > self._activation_threshold

        # Handle activation state changes
        if is_active and not self._is_active:
            # Just activated - store reference pose
            self._ref_controller_pos = controller_pos.copy()
            self._ref_controller_quat = controller_quat.copy()
            self._is_active = True
            # Return zero delta on first activation
            return torch.zeros(6, dtype=torch.float32, device=self._sim_device)
        elif not is_active:
            # Not active - reset and return zero
            self._is_active = False
            self._ref_controller_pos = None
            self._ref_controller_quat = None
            self._smoothed_delta_pos = np.zeros(3)
            self._smoothed_delta_rot = np.zeros(3)
            return torch.zeros(6, dtype=torch.float32, device=self._sim_device)

        # Calculate delta pose
        delta_command = self._calculate_delta_pose(controller_pos, controller_quat)

        # Convert to torch tensor
        ee_command = torch.tensor(delta_command, dtype=torch.float32, device=self._sim_device)

        return ee_command

    def _calculate_delta_pose(self, controller_pos: np.ndarray, controller_quat: np.ndarray) -> np.ndarray:
        """Calculate delta pose from reference controller pose.

        Args:
            controller_pos: Current controller position [x, y, z]
            controller_quat: Current controller quaternion [qw, qx, qy, qz]

        Returns:
            np.ndarray: 6D array with position delta (xyz) and rotation delta as axis-angle (rx,ry,rz)
        """
        # Apply coordinate transformation
        controller_pos_world = self._R_xr_to_world @ controller_pos
        ref_pos_world = self._R_xr_to_world @ self._ref_controller_pos

        # Calculate position delta
        delta_pos = controller_pos_world - ref_pos_world

        # Calculate rotation delta
        # Convert quaternions to scipy Rotation objects (expects [qx, qy, qz, qw])
        current_rot = Rotation.from_quat([controller_quat[1], controller_quat[2], controller_quat[3], controller_quat[0]])
        ref_rot = Rotation.from_quat([self._ref_controller_quat[1], self._ref_controller_quat[2],
                                      self._ref_controller_quat[3], self._ref_controller_quat[0]])

        # Calculate relative rotation
        relative_rotation = current_rot * ref_rot.inv()
        delta_rot = relative_rotation.as_rotvec()

        # Apply rotation transformation
        R_transform = np.eye(4)
        R_transform[:3, :3] = self._R_xr_to_world
        R_quat = Rotation.from_matrix(self._R_xr_to_world)
        delta_rot_world = (R_quat * Rotation.from_rotvec(delta_rot) * R_quat.inv()).as_rotvec()

        # Apply zero_out_xy_rotation if enabled
        if self._zero_out_xy_rotation:
            delta_rot_world[0] = 0  # x-axis
            delta_rot_world[1] = 0  # y-axis

        # Smooth and scale position
        self._smoothed_delta_pos = self._alpha_pos * delta_pos + (1 - self._alpha_pos) * self._smoothed_delta_pos
        if np.linalg.norm(self._smoothed_delta_pos) < self._position_threshold:
            self._smoothed_delta_pos = np.zeros(3)
        position = self._smoothed_delta_pos * self._pos_scale_factor

        # Smooth and scale rotation
        self._smoothed_delta_rot = self._alpha_rot * delta_rot_world + (1 - self._alpha_rot) * self._smoothed_delta_rot
        if np.linalg.norm(self._smoothed_delta_rot) < self._rotation_threshold:
            self._smoothed_delta_rot = np.zeros(3)
        rotation = self._smoothed_delta_rot * self._rot_scale_factor

        # Update visualization if enabled
        if self._enable_visualization:
            # Convert rotation vector to quaternion and combine with current rotation
            delta_quat = Rotation.from_rotvec(rotation).as_quat()  # x, y, z, w format
            current_viz_rot = Rotation.from_quat([self._visualization_rot[1], self._visualization_rot[2],
                                                   self._visualization_rot[3], self._visualization_rot[0]])
            new_rot = Rotation.from_quat(delta_quat) * current_viz_rot
            self._visualization_pos = self._visualization_pos + position
            # Convert back to w, x, y, z format
            new_rot_quat = new_rot.as_quat()
            self._visualization_rot = np.array([new_rot_quat[3], new_rot_quat[0], new_rot_quat[1], new_rot_quat[2]])
            self._update_visualization()

        return np.concatenate([position, rotation])

    def _update_visualization(self):
        """Update visualization markers with current pose."""
        if self._enable_visualization:
            trans = torch.tensor([self._visualization_pos], dtype=torch.float32, device=self._sim_device)
            # Visualization expects [w, x, y, z] format
            rot = torch.tensor([self._visualization_rot], dtype=torch.float32, device=self._sim_device)
            self._goal_marker.visualize(translations=trans, orientations=rot)