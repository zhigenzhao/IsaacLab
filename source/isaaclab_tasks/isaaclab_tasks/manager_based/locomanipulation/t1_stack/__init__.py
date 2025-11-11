# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T1 humanoid cube stacking tasks for Isaac Lab."""

import gymnasium as gym

from . import t1_stack_env_cfg, t1_stack_mimic_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-T1-Stack-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": t1_stack_env_cfg.T1CubeStackEnvCfg,
    },
)

gym.register(
    id="Isaac-T1-Stack-Cube-Mimic-v0",
    entry_point="isaaclab_tasks.manager_based.locomanipulation.t1_stack.t1_stack_mimic_env:T1StackMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": t1_stack_mimic_env_cfg.T1StackMimicEnvCfg,
    },
)
