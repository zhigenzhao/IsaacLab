# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functions for synchronizing teleoperation device state with simulation."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def sync_teleop_device_state_on_reset(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """Synchronize teleoperation device state with simulation on environment reset.

    This function updates the internal state of teleop devices (particularly IK retargeters)
    to match the current robot joint positions in the simulation. This prevents drift and
    ensures consistent behavior after resets.

    Args:
        env: The environment instance
        env_ids: IDs of environments being reset
        asset_cfg: Configuration specifying which asset to read joint positions from
    """
    # Only sync for env 0 (single device for all envs)
    if 0 not in env_ids:
        return

    # Check if environment has teleop devices configured
    if not hasattr(env.cfg, "teleop_devices") or env.cfg.teleop_devices is None:
        return

    # Get the robot asset
    from isaaclab.assets import Articulation
    asset: Articulation = env.scene[asset_cfg.name]

    # Get upper body joint positions for env 0
    upper_body_joint_names = [
        "AAHead_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
    ]

    try:
        upper_body_joint_ids = asset.find_joints(upper_body_joint_names, preserve_order=True)[0]
        measured_joint_pos = asset.data.joint_pos[0, upper_body_joint_ids]

        # Get device instances from teleop_device_factory or environment
        # Note: This requires the environment to store device instances
        # For now, we'll store them in a way that's accessible
        if hasattr(env, '_teleop_devices'):
            for device_name, device in env._teleop_devices.items():
                # Reset device state
                if hasattr(device, 'set_measured_joint_positions'):
                    device.set_measured_joint_positions(measured_joint_pos)

                # Reset retargeters
                if hasattr(device, '_retargeters'):
                    for retargeter in device._retargeters:
                        if hasattr(retargeter, 'reset'):
                            retargeter.reset(measured_joint_pos.cpu().numpy())

    except Exception as e:
        # Silently fail if joints not found or device not available
        pass


def update_teleop_device_state_pre_step(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """Update teleoperation device with current measured joint positions before step.

    This function provides the current robot joint positions to teleop devices so they
    can synchronize their internal IK solver state with the actual simulation state.

    Args:
        env: The environment instance
        asset_cfg: Configuration specifying which asset to read joint positions from
    """
    # Check if environment has teleop devices configured
    if not hasattr(env.cfg, "teleop_devices") or env.cfg.teleop_devices is None:
        return

    # Get the robot asset
    from isaaclab.assets import Articulation
    asset: Articulation = env.scene[asset_cfg.name]

    # Get upper body joint positions for env 0
    upper_body_joint_names = [
        "AAHead_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
    ]

    try:
        upper_body_joint_ids = asset.find_joints(upper_body_joint_names, preserve_order=True)[0]
        measured_joint_pos = asset.data.joint_pos[0, upper_body_joint_ids]

        # Update device state
        if hasattr(env, '_teleop_devices'):
            for device_name, device in env._teleop_devices.items():
                if hasattr(device, 'set_measured_joint_positions'):
                    device.set_measured_joint_positions(measured_joint_pos)

    except Exception as e:
        # Silently fail if joints not found or device not available
        pass
