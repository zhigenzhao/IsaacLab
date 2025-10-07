# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T1 humanoid teleoperation tasks for Isaac Lab."""

import gymnasium as gym

from . import t1_teleop_env_cfg, t1_teleop_manipulation_env_cfg, t1_teleop_reach_env_cfg
from .tasks import t1_teleop_walk_to_target_cfg, t1_teleop_transport_box

##
# Register Gym environments.
##

gym.register(
    id="Isaac-T1-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": t1_teleop_env_cfg.T1TeleopEnvCfg,
    },
)

gym.register(
    id="Isaac-T1-Teleop-WalkToTarget-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": t1_teleop_walk_to_target_cfg.T1TeleopEnvCfg_WalkToTarget,
    },
)

gym.register(
    id="Isaac-T1-Teleop-Manipulation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": t1_teleop_manipulation_env_cfg.T1TeleopManipulationEnvCfg,
    },
)

gym.register(
    id="Isaac-T1-Teleop-Reach",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": t1_teleop_reach_env_cfg.T1TeleopReachEnvCfg,
    },
)