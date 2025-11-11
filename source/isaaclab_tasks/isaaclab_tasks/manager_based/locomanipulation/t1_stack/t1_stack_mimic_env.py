# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T1 Stack Mimic Environment for automatic annotation and data generation."""

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import SceneEntityCfg

from . import t1_stack_mdp


class T1StackMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for T1 cube stacking task.

    This environment enables automatic annotation of demonstrations and dataset generation
    for imitation learning. It wraps the T1 stack environment with mimic-specific methods
    for tracking end-effector poses and detecting subtask completions.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose (hand base link).

        Args:
            eef_name: Name of the end effector ("left" or "right").
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        robot: Articulation = self.scene["robot"]

        # Get hand base link position and orientation
        hand_link = f"{eef_name}_base_link"
        body_ids = robot.find_bodies(hand_link)[0]
        hand_pos = robot.data.body_pos_w[env_ids, body_ids]
        hand_quat = robot.data.body_quat_w[env_ids, body_ids]

        # Convert to 4x4 pose matrix
        return PoseUtils.make_pose(hand_pos, PoseUtils.matrix_from_quat(hand_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector and returns an action.

        Note: For T1 with joint position control, this is a simplified implementation.
        Full IK solution would be needed for production use.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            action_noise_dict: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to compute the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        robot: Articulation = self.scene["robot"]

        # Get current joint positions as baseline
        current_joint_pos = robot.data.joint_pos[env_id].clone()

        # For now, return current joint positions with gripper actions
        # A full implementation would use IK to compute joint positions from target poses
        left_gripper = gripper_action_dict.get("left", torch.tensor([0.0], device=self.device))
        right_gripper = gripper_action_dict.get("right", torch.tensor([0.0], device=self.device))

        # Action is: upper_joint_pos (16) + left_gripper (1) + right_gripper (1)
        from ..t1_common.joint_names import T1_UPPER_BODY_JOINTS
        upper_joint_indices = robot.find_joints(T1_UPPER_BODY_JOINTS, preserve_order=True)[0]
        upper_joint_pos = current_joint_pos[upper_joint_indices]

        action = torch.cat([upper_joint_pos, left_gripper, right_gripper], dim=0)

        # Add noise if specified
        if action_noise_dict is not None:
            noise = torch.randn_like(action) * 0.01  # Small noise for joint positions
            action += noise
            action = torch.clamp(action, -3.0, 3.0)  # Clamp to reasonable joint limits

        return action

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action to target pose for the end effector.

        For T1 with joint position control, this uses forward kinematics to compute
        the resulting hand poses from the joint position action.

        Args:
            action: Environment action. Shape is (num_envs, action_dim)

        Returns:
            A dictionary of eef pose torch.Tensor that action corresponds to.
        """
        # After applying the action, the hands will move to new poses
        # We compute what those poses would be using the robot's current state
        # This is a simplified implementation - full FK would be more accurate

        target_poses = {}
        for eef_name in ["left", "right"]:
            # Get current hand pose (this would be the result after stepping with the action)
            target_poses[eef_name] = self.get_robot_eef_pose(eef_name, env_ids=None)

        return target_poses

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions.

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        # Action format: upper_joint_pos (16) + left_gripper (1) + right_gripper (1)
        # Total: 18 dimensions

        # Handle both 2D (timesteps, action_dim) and 3D (num_envs, timesteps, action_dim)
        if actions.ndim == 2:
            left_gripper = actions[:, 16:17]  # Index 16
            right_gripper = actions[:, 17:18]  # Index 17
        elif actions.ndim == 3:
            left_gripper = actions[:, :, 16:17]  # Index 16
            right_gripper = actions[:, :, 17:18]  # Index 17
        else:
            raise ValueError(f"Expected actions to be 2D or 3D, got shape {actions.shape}")

        return {
            "left": left_gripper,
            "right": right_gripper
        }

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in the task.

        For T1 stack task, we detect two subtasks:
        1. Grasp cube_2 (red cube) with either hand
        2. Stack cube_2 on top of cube_1 (blue cube)

        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.

        Returns:
            A dictionary termination signal flags (False or True) for each subtask.
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = {}

        # Subtask 1: Grasp cube_2 with either hand
        left_grasp = t1_stack_mdp.object_grasped_by_hand(
            self,
            hand="left",
            robot_cfg=SceneEntityCfg("robot"),
            object_cfg=SceneEntityCfg("cube_2"),
        )[env_ids]

        right_grasp = t1_stack_mdp.object_grasped_by_hand(
            self,
            hand="right",
            robot_cfg=SceneEntityCfg("robot"),
            object_cfg=SceneEntityCfg("cube_2"),
        )[env_ids]

        # Grasp is complete if either hand has grasped the cube
        signals["grasp_cube_2"] = ((left_grasp > 0.5) | (right_grasp > 0.5)).float()

        # Subtask 2: Stack cube_2 on cube_1
        stack_complete = t1_stack_mdp.object_stacked(
            self,
            upper_object_cfg=SceneEntityCfg("cube_2"),
            lower_object_cfg=SceneEntityCfg("cube_1"),
        )[env_ids]

        signals["stack_cube_2_on_cube_1"] = (stack_complete > 0.5).float()

        return signals
