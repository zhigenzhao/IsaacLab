# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T1 humanoid box lifting tasks for Isaac Lab."""

import gymnasium as gym

from . import t1_lift_box_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-T1-Lift-Box-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": t1_lift_box_env_cfg.T1LiftBoxEnvCfg,
    },
)
