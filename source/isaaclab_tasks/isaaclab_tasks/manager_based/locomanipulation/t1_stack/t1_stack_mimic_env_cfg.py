# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T1 Stack Mimic Environment Configuration for automatic annotation and data generation."""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig, DataGenConfig
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from . import t1_stack_mdp
from .t1_stack_env_cfg import T1CubeStackEnvCfg


@configclass
class T1StackMimicEnvCfg(T1CubeStackEnvCfg, MimicEnvCfg):
    """Configuration for the T1 cube stacking mimic environment.

    This configuration extends the base T1 stacking environment with mimic-specific
    settings for automatic annotation and dataset generation.
    """

    def __post_init__(self):
        """Post initialization to set up mimic-specific configurations."""
        # Call parent post_init first
        super().__post_init__()

        # Configure data generation settings
        self.datagen_config = DataGenConfig()
        self.datagen_config.name = "t1_stack"
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # Define subtask configurations for both hands
        # Since T1 can use either left or right hand for manipulation,
        # we create subtask configs for both end-effectors

        # Left hand subtasks
        left_subtasks = []
        left_subtasks.append(
            SubTaskConfig(
                # Grasp cube_2 (red cube) with left hand
                object_ref="cube_2",
                subtask_term_signal="grasp_cube_2",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp red cube with left hand",
                next_subtask_description="Stack red cube on top of blue cube",
            )
        )
        left_subtasks.append(
            SubTaskConfig(
                # Stack cube_2 on cube_1
                object_ref="cube_1",
                # End of final subtask does not need to be detected
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Stack red cube on blue cube",
            )
        )

        # Right hand subtasks (mirror of left hand)
        right_subtasks = []
        right_subtasks.append(
            SubTaskConfig(
                # Grasp cube_2 (red cube) with right hand
                object_ref="cube_2",
                subtask_term_signal="grasp_cube_2",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp red cube with right hand",
                next_subtask_description="Stack red cube on top of blue cube",
            )
        )
        right_subtasks.append(
            SubTaskConfig(
                # Stack cube_2 on cube_1
                object_ref="cube_1",
                # End of final subtask does not need to be detected
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Stack red cube on blue cube",
            )
        )

        # Assign subtask configs for both end-effectors
        self.subtask_configs = {
            "left": left_subtasks,
            "right": right_subtasks,
        }

        # No subtask constraints for single-arm manipulation
        self.task_constraint_configs = []

        # Add observation terms for mimic subtask signals
        # These match the subtask_term_signal names and are used by the UI
        self.observations.subtask_terms.grasp_cube_2 = ObsTerm(func=t1_stack_mdp.grasp_cube_2)
        self.observations.subtask_terms.stack_cube_2_on_cube_1 = ObsTerm(func=t1_stack_mdp.stack_cube_2_on_cube_1)
