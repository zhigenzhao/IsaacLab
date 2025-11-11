# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP functions specific to T1 cube stacking task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Observation functions
##


def cube_positions_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Cube positions in robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]

    # Get robot base pose
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    # Transform cube positions to robot frame
    cube_1_pos_b, _ = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w, cube_1.data.root_pos_w, cube_1.data.root_quat_w
    )
    cube_2_pos_b, _ = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w, cube_2.data.root_pos_w, cube_2.data.root_quat_w
    )

    return torch.cat((cube_1_pos_b, cube_2_pos_b), dim=1)


def cube_orientations_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Cube orientations in robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]

    # Get robot base pose
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    # Transform cube orientations to robot frame
    _, cube_1_quat_b = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w, cube_1.data.root_pos_w, cube_1.data.root_quat_w
    )
    _, cube_2_quat_b = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w, cube_2.data.root_pos_w, cube_2.data.root_quat_w
    )

    return torch.cat((cube_1_quat_b, cube_2_quat_b), dim=1)


def object_grasped_by_hand(
    env: ManagerBasedRLEnv,
    hand: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    distance_threshold: float = 0.15,
    gripper_closed_min: float = 0.15,
    gripper_closed_max: float = 0.35,
) -> torch.Tensor:
    """Check if object is grasped by specified hand.

    Args:
        hand: "left" or "right"
        distance_threshold: Maximum distance between hand and object (meters)
        gripper_closed_min: Minimum gripper joint position for grasping (rad)
        gripper_closed_max: Maximum gripper joint position for grasping (rad)
            The gripper won't fully close when holding an object, so we check
            if it's in the "grasping range" between partially and mostly closed.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Get hand position from articulation body data
    # The T1 gripper uses "left_base_link" and "right_base_link" as the hand bodies
    hand_link = f"{hand}_base_link"
    body_ids = robot.find_bodies(hand_link)[0]
    # Ensure we get a single index (find_bodies returns list of indices)
    body_id = body_ids[0] if hasattr(body_ids, '__getitem__') and len(body_ids) > 0 else body_ids
    hand_pos = robot.data.body_pos_w[:, body_id]

    # Get gripper joint position
    gripper_joint = f"{hand}_Link1"
    joint_ids = robot.find_joints(gripper_joint)[0]
    # Ensure we get a single index (find_joints returns list of indices)
    joint_id = joint_ids[0] if hasattr(joint_ids, '__getitem__') and len(joint_ids) > 0 else joint_ids
    gripper_pos = robot.data.joint_pos[:, joint_id]

    # Check distance (hand_pos should be shape [num_envs, 3], obj_pos is [num_envs, 3])
    obj_pos = obj.data.root_pos_w
    distance = torch.norm(hand_pos - obj_pos, dim=-1)

    # Check if gripper is in the "grasping range" and object is close
    # The gripper should be partially closed (holding the object), not fully closed (empty)
    # distance is [num_envs], gripper_pos is [num_envs]
    is_close = distance < distance_threshold
    is_in_grasp_range = (gripper_pos > gripper_closed_min) & (gripper_pos < gripper_closed_max)
    is_grasped = is_close & is_in_grasp_range

    # Return [num_envs, 1]
    return is_grasped.unsqueeze(-1).float()


def object_stacked(
    env: ManagerBasedRLEnv,
    upper_object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    lower_object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    height_tolerance: float = 0.02,
    lateral_tolerance: float = 0.03,
) -> torch.Tensor:
    """Check if upper object is stacked on lower object.

    Args:
        height_tolerance: Tolerance for height check (meters)
        lateral_tolerance: Tolerance for x-y alignment (meters)
    """
    upper: RigidObject = env.scene[upper_object_cfg.name]
    lower: RigidObject = env.scene[lower_object_cfg.name]

    # Expected height difference (cube side length is ~0.0406m)
    expected_height_diff = 0.0406

    # Check height alignment
    height_diff = upper.data.root_pos_w[:, 2] - lower.data.root_pos_w[:, 2]
    height_aligned = torch.abs(height_diff - expected_height_diff) < height_tolerance

    # Check lateral alignment (x-y plane)
    lateral_diff = torch.norm(
        upper.data.root_pos_w[:, :2] - lower.data.root_pos_w[:, :2], dim=1
    )
    lateral_aligned = lateral_diff < lateral_tolerance

    is_stacked = height_aligned & lateral_aligned

    return is_stacked.unsqueeze(-1).float()


def grasp_cube_2(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Check if cube_2 is grasped by either hand (for mimic subtask signal).

    This is used as an observation term for the mimic environment to track
    the 'grasp_cube_2' subtask completion.
    """
    # Check if either hand has grasped cube_2
    left_grasp = object_grasped_by_hand(env, hand="left", robot_cfg=robot_cfg, object_cfg=object_cfg)
    right_grasp = object_grasped_by_hand(env, hand="right", robot_cfg=robot_cfg, object_cfg=object_cfg)

    # Return True if either hand has grasped
    # Both return shape (num_envs, 1), so squeeze and combine
    is_grasped = ((left_grasp.squeeze(-1) > 0.5) | (right_grasp.squeeze(-1) > 0.5)).unsqueeze(-1)
    return is_grasped.float()


def stack_cube_2_on_cube_1(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Check if cube_2 is stacked on cube_1 (for mimic subtask signal).

    This is used as an observation term for the mimic environment to track
    the 'stack_cube_2_on_cube_1' subtask completion.
    """
    return object_stacked(env, upper_object_cfg=cube_2_cfg, lower_object_cfg=cube_1_cfg)


##
# Termination functions
##


def cubes_stacked(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Check if cube_2 (red) is stacked on cube_1 (blue)."""
    # Check if cube_2 is on cube_1
    stack_1 = object_stacked(env, cube_2_cfg, cube_1_cfg)

    # Success if stack is complete
    success = stack_1 > 0.5

    return success.squeeze(-1)


##
# Event functions
##


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


def randomize_object_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    pose_range: dict,
    min_separation: float = 0.1,
):
    """Randomize object poses with minimum separation constraint.

    Args:
        pose_range: Dictionary with keys 'x', 'y', 'z', 'yaw' containing (min, max) tuples
        min_separation: Minimum distance between objects (meters)
    """
    num_resets = len(env_ids)

    for i, asset_cfg in enumerate(asset_cfgs):
        asset: RigidObject = env.scene[asset_cfg.name]

        # Generate random positions with separation constraint
        max_attempts = 100
        for attempt in range(max_attempts):
            # Sample random position
            x = torch.rand(num_resets, device=env.device) * (pose_range["x"][1] - pose_range["x"][0]) + pose_range["x"][0]
            y = torch.rand(num_resets, device=env.device) * (pose_range["y"][1] - pose_range["y"][0]) + pose_range["y"][0]
            z = torch.full((num_resets,), pose_range["z"][0], device=env.device)

            # Sample random yaw
            yaw = torch.rand(num_resets, device=env.device) * (pose_range["yaw"][1] - pose_range["yaw"][0]) + pose_range["yaw"][0]

            # Check separation from previously placed objects
            valid = torch.ones(num_resets, dtype=torch.bool, device=env.device)
            for j in range(i):
                prev_asset: RigidObject = env.scene[asset_cfgs[j].name]
                prev_pos = prev_asset.data.root_pos_w[env_ids, :2]
                curr_pos = torch.stack([x, y], dim=1)
                distance = torch.norm(curr_pos - prev_pos, dim=1)
                valid &= (distance >= min_separation)

            if valid.all() or attempt == max_attempts - 1:
                break

        # Convert yaw to quaternion
        quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )

        # Set pose
        pos = torch.stack([x, y, z], dim=1)
        asset.write_root_pose_to_sim(
            torch.cat([pos, quat], dim=1), env_ids=env_ids
        )

        # Reset velocity
        asset.write_root_velocity_to_sim(
            torch.zeros((num_resets, 6), device=env.device), env_ids=env_ids
        )
