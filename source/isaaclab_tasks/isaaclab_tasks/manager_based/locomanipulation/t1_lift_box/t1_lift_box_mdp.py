# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP functions specific to T1 box lifting task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Observation functions
##


def box_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Box position in robot base frame.

    Returns:
        torch.Tensor: Box position in robot frame, shape [num_envs, 3]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]

    # Get robot base pose
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    # Transform box position to robot frame
    box_pos_b, _ = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w, box.data.root_pos_w, box.data.root_quat_w
    )

    return box_pos_b


def box_orientation_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Box orientation in robot base frame.

    Returns:
        torch.Tensor: Box quaternion (w, x, y, z) in robot frame, shape [num_envs, 4]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]

    # Get robot base pose
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    # Transform box orientation to robot frame
    _, box_quat_b = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w, box.data.root_pos_w, box.data.root_quat_w
    )

    return box_quat_b


##
# Termination functions
##


def box_lifted_success(
    env: ManagerBasedRLEnv,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    height_threshold: float = 0.75,
) -> torch.Tensor:
    """Check if the box's z-height is strictly above height_threshold.

    Note: The hold duration logic is handled externally by the recording script
    via --num_success_steps argument, which counts consecutive successful steps.

    Args:
        env: The environment instance
        box_cfg: Configuration for the box scene entity
        height_threshold: Minimum height for success (meters)

    Returns:
        torch.Tensor: Boolean tensor of shape [num_envs] indicating if box is above threshold
    """
    box: RigidObject = env.scene[box_cfg.name]

    # Current box z height
    z = box.data.root_pos_w[:, 2]
    above = z > height_threshold

    return above


def box_dropped(
    env: ManagerBasedRLEnv,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    height_threshold: float = 0.2,
) -> torch.Tensor:
    """Fail when the box's z-height is below height_threshold.

    Args:
        env: The environment instance
        box_cfg: Configuration for the box scene entity
        height_threshold: Height below which box is considered dropped (meters)

    Returns:
        torch.Tensor: Boolean tensor of shape [num_envs] indicating failure
    """
    box: RigidObject = env.scene[box_cfg.name]
    z = box.data.root_pos_w[:, 2]
    return z < height_threshold


##
# Event functions
##


def randomize_box_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    box_cfg: SceneEntityCfg,
    table_cfg: SceneEntityCfg,
    base_pos_xy: tuple[float, float] = (0.5, 0.0),
    yaw_range_deg: float = 15.0,
):
    """Randomize box position and orientation on table.

    Box position randomization:
    - X: base_pos_xy[0] + uniform(-0.08, 0.02)
    - Y: base_pos_xy[1] + uniform(-0.06, 0.06)
    - Z: 0.62 (table surface height from USD)
    - Yaw: uniform(-yaw_range_deg, +yaw_range_deg)

    Table is placed at fixed position with 90° rotation.

    Args:
        env: The environment instance
        env_ids: Environment indices to reset
        box_cfg: Configuration for the box scene entity
        table_cfg: Configuration for the table scene entity
        base_pos_xy: Base position (x, y) for table and box
        yaw_range_deg: Range of yaw randomization in degrees
    """
    dev = env.device
    ids = env_ids.to(dev)
    n = ids.numel()

    box: RigidObject = env.scene[box_cfg.name]
    table: RigidObject = env.scene[table_cfg.name]

    # Environment origins (grid centers)
    env_origins = env.scene.env_origins[ids].to(dev)

    # Table position (fixed)
    x_table, y_table = base_pos_xy
    z_table = 0.0
    z_box = 0.62  # Table surface height from USD

    pos_table = env_origins.clone()
    pos_table[:, 0] += x_table
    pos_table[:, 1] += y_table
    pos_table[:, 2] = z_table

    # Box position randomization
    x_box_noise_range = (-0.08, 0.02)
    y_box_noise_range = (-0.06, 0.06)

    x_lo, x_hi = x_box_noise_range
    y_lo, y_hi = y_box_noise_range

    dx = x_lo + (x_hi - x_lo) * torch.rand((n, 1), device=dev)
    dy = y_lo + (y_hi - y_lo) * torch.rand((n, 1), device=dev)

    pos_box = env_origins.clone()
    pos_box[:, 0:1] += base_pos_xy[0] + dx
    pos_box[:, 1:2] += base_pos_xy[1] + dy
    pos_box[:, 2] = z_box

    # Box yaw randomization
    yaw_lo_deg, yaw_hi_deg = -yaw_range_deg, +yaw_range_deg
    yaw_deg_rand = yaw_lo_deg + (yaw_hi_deg - yaw_lo_deg) * torch.rand((n,), device=dev)
    yaw = torch.deg2rad(yaw_deg_rand)
    roll = torch.zeros_like(yaw)
    pitch = torch.zeros_like(yaw)
    quat_box = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

    # Table orientation: 90-degree rotation around z-axis
    angle_table = torch.full((n,), math.pi / 2, device=dev, dtype=torch.float32)
    axis_table = torch.zeros((n, 3), device=dev, dtype=torch.float32)
    axis_table[:, 2] = 1.0  # z-axis
    quat_table = math_utils.quat_from_angle_axis(angle_table, axis_table)

    vel = torch.zeros((n, 6), device=dev)

    pose_box = torch.cat([pos_box, quat_box], dim=-1)
    pose_table = torch.cat([pos_table, quat_table], dim=-1)

    # Write pose and zero velocity to PhysX
    box.write_root_pose_to_sim(pose_box, env_ids=ids)
    box.write_root_velocity_to_sim(vel, env_ids=ids)

    table.write_root_pose_to_sim(pose_table, env_ids=ids)
    table.write_root_velocity_to_sim(vel, env_ids=ids)


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    mean: float = 0.0,
    std: float = 0.01,
):
    """Randomize joint positions by adding Gaussian noise to current position.

    NOTE: This function adds noise to the CURRENT joint positions (after any previous
    reset events), not the default joint positions. This ensures that if reset_to_prep()
    runs before this function, the randomization adds noise around the prep pose rather
    than overwriting it with near-zero values.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get current joint positions (not default!) to add noise to the state
    # set by previous reset events like reset_to_prep()
    joint_pos = asset.data.joint_pos[env_ids].clone()

    # Add Gaussian noise
    noise = torch.randn_like(joint_pos) * std + mean
    joint_pos += noise

    # Clamp to joint limits
    joint_pos = joint_pos.clamp(
        asset.data.soft_joint_pos_limits[env_ids, :, 0],
        asset.data.soft_joint_pos_limits[env_ids, :, 1],
    )

    # Set joint positions (keep velocities at zero for reset)
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(asset.data.joint_vel[env_ids]), env_ids=env_ids)
