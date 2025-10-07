# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab.envs import ManagerBasedRLEnv
import torch
import isaaclab.envs.mdp as mdp

PHASE = 1.0 / 1.2


def get_phase(env: ManagerBasedRLEnv, phase_dt: float):
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float)

    phase = (
        torch.fmod(
            env.episode_length_buf.type(dtype=torch.float) * env.step_dt,
            phase_dt,
        )
        / phase_dt
    )
    return phase


def should_stand(env: ManagerBasedRLEnv, zero_threshold: float = 0.1) -> torch.Tensor:
    command = env.command_manager.get_command("base_velocity")
    return torch.norm(command, dim=1) < zero_threshold


def should_walk(env: ManagerBasedRLEnv, zero_threshold: float = 0.1) -> torch.Tensor:
    command = env.command_manager.get_command("base_velocity")
    return torch.norm(command, dim=1) >= zero_threshold


def conditioned_get_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    phase = get_phase(env, PHASE)
    standing = should_stand(env)
    return torch.where(standing, torch.zeros_like(phase), phase)


def clock(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Clock time using sin and cos from the phase of the simulation."""
    phase = conditioned_get_phase(env)
    return torch.cat(
        [
            torch.sin(2 * torch.pi * phase).unsqueeze(1),
            torch.cos(2 * torch.pi * phase).unsqueeze(1),
        ],
        dim=1,
    ).to(env.device)


@configclass
class ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        clock = ObsTerm(
            func=clock,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=1,
            noise=Gnoise(mean=0.0, std=0.1),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Gnoise(mean=0.0, std=0.01),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            scale=1,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            scale=1.0,
            noise=Gnoise(mean=0.0, std=0.01),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "AAHead_yaw",
                        "Head_pitch",
                        "Left_Shoulder_Pitch",
                        "Left_Shoulder_Roll",
                        "Left_Elbow_Pitch",
                        "Left_Elbow_Yaw",
                        "Left_Wrist_Pitch",
                        "Left_Wrist_Yaw",
                        "Left_Hand_Roll",
                        "Right_Shoulder_Pitch",
                        "Right_Shoulder_Roll",
                        "Right_Elbow_Pitch",
                        "Right_Elbow_Yaw",
                        "Right_Wrist_Pitch",
                        "Right_Wrist_Yaw",
                        "Right_Hand_Roll",
                        "Waist",
                        "Left_Hip_Pitch",
                        "Left_Hip_Roll",
                        "Left_Hip_Yaw",
                        "Left_Knee_Pitch",
                        "Left_Ankle_Pitch",
                        "Left_Ankle_Roll",
                        "Right_Hip_Pitch",
                        "Right_Hip_Roll",
                        "Right_Hip_Yaw",
                        "Right_Knee_Pitch",
                        "Right_Ankle_Pitch",
                        "Right_Ankle_Roll",
                    ],
                    preserve_order=True,
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            scale=1.0,
            noise=Gnoise(mean=0.0, std=0.05),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "AAHead_yaw",
                        "Head_pitch",
                        "Left_Shoulder_Pitch",
                        "Left_Shoulder_Roll",
                        "Left_Elbow_Pitch",
                        "Left_Elbow_Yaw",
                        "Left_Wrist_Pitch",
                        "Left_Wrist_Yaw",
                        "Left_Hand_Roll",
                        "Right_Shoulder_Pitch",
                        "Right_Shoulder_Roll",
                        "Right_Elbow_Pitch",
                        "Right_Elbow_Yaw",
                        "Right_Wrist_Pitch",
                        "Right_Wrist_Yaw",
                        "Right_Hand_Roll",
                        "Waist",
                        "Left_Hip_Pitch",
                        "Left_Hip_Roll",
                        "Left_Hip_Yaw",
                        "Left_Knee_Pitch",
                        "Left_Ankle_Pitch",
                        "Left_Ankle_Roll",
                        "Right_Hip_Pitch",
                        "Right_Hip_Roll",
                        "Right_Hip_Yaw",
                        "Right_Knee_Pitch",
                        "Right_Ankle_Pitch",
                        "Right_Ankle_Roll",
                    ],
                    preserve_order=True,
                )
            },
        )
        actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "lower_joint_pos"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 8

    # observation groups
    policy: PolicyCfg = PolicyCfg()
