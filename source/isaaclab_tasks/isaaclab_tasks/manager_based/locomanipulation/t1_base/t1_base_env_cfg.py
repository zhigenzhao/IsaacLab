# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for T1 base environment with position control.

This is a minimal base environment for the T1 humanoid robot using position control.
It is designed for whole-body teleoperation via joint state tracking policies (e.g., TWIST).

Action space (31 DOFs):
- body_joint_pos: 29 DOFs (full body from TWIST retargeter)
- left_gripper: 1 DOF (left gripper from XR trigger)
- right_gripper: 1 DOF (right gripper from XR trigger)

Note: The robot has 37 total DOFs (29 body + 8 gripper), but only 31 are directly controlled.
The 6 gripper mimic joints follow the main gripper joints automatically.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.devices import DevicesCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.envs.mdp.observations import (
    base_pos_z,
    base_lin_vel,
    base_ang_vel,
    projected_gravity,
    joint_pos,
    joint_vel,
    last_action,
)
from isaaclab.envs.mdp.terminations import time_out, root_height_below_minimum
from isaaclab.envs.mdp.events import reset_root_state_uniform, reset_joints_by_offset
from isaaclab.envs.mdp.rewards import is_alive
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.booster import T1_W_GRIPPER_CFG
from isaaclab_tasks.manager_based.locomanipulation.t1_common.joint_names import T1_FULL_BODY_JOINTS
from isaaclab_tasks.manager_based.locomanipulation.t1_common.xr_controller_cfg import create_t1_xr_twist_cfg

# Default path to TWIST config file (relative to t1_common module)
_DEFAULT_TWIST_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "t1_common", "twist_t1_xrt.yaml"
)


##
# Scene definition
##


@configclass
class T1BaseSceneCfg(InteractiveSceneCfg):
    """Configuration for the T1 base scene on flat ground."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot (37 DOF: 29 body + 8 gripper)
    robot = T1_W_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Action groups are structured to match XR teleop output order:
    - body_joint_pos: 29 DOFs from TWIST retargeter (full body)
    - left_gripper: 1 DOF from left gripper retargeter
    - right_gripper: 1 DOF from right gripper retargeter
    Total: 31 DOFs
    """

    # Full body joints (29 DOFs) - controlled by TWIST retargeter
    body_joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=T1_FULL_BODY_JOINTS,
        scale=1.0,
        preserve_order=True,
        use_default_offset=False,
    )

    # Left gripper (1 DOF) - controlled by left gripper retargeter
    left_gripper = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_Link1"],
        scale=1.0,
        preserve_order=True,
        use_default_offset=False,
    )

    # Right gripper (1 DOF) - controlled by right gripper retargeter
    right_gripper = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_Link1"],
        scale=1.0,
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # Robot base state
        base_height = ObsTerm(func=base_pos_z)
        base_lin_vel = ObsTerm(func=base_lin_vel)
        base_ang_vel = ObsTerm(func=base_ang_vel)
        projected_gravity = ObsTerm(func=projected_gravity)

        # Joint state
        joint_pos = ObsTerm(func=joint_pos)
        joint_vel = ObsTerm(func=joint_vel)

        # Action history
        actions = ObsTerm(func=last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Stay alive bonus
    alive = RewTerm(func=is_alive, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Terminate if the episode length is exceeded
    time_out = DoneTerm(func=time_out, time_out=True)

    # Terminate if the robot falls (T1 standing height is ~0.7m)
    torso_height = DoneTerm(func=root_height_below_minimum, params={"minimum_height": 0.4})


@configclass
class T1BaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the T1 base environment with position control."""

    # Scene settings
    scene: T1BaseSceneCfg = T1BaseSceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2

        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0

        # Configure teleoperation devices for XR full-body control with TWIST
        # Users can override twist_config_path via environment variable or by modifying this config
        twist_config_path = os.environ.get("TWIST_CONFIG_PATH", _DEFAULT_TWIST_CONFIG)
        self.teleop_devices = DevicesCfg(
            devices={
                "xr_controller_full_body": create_t1_xr_twist_cfg(
                    twist_config_path=twist_config_path,
                    sim_device=self.sim.device,
                ),
            }
        )
