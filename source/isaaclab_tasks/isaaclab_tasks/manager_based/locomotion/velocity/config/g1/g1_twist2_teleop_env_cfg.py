# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 TWIST2 teleoperation environment configuration.

This environment is designed for whole-body teleoperation of the Unitree G1 humanoid
robot using XR controllers and the TWIST2 imitation learning policy.

Total action space: 29 DOFs (body joints only)
"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab.assets import Articulation

# Import robot configuration (regular G1 29DOF without inspire hands)
from isaaclab_assets import G1_29DOF_CFG

# Import teleop device configurations
from isaaclab.devices import DevicesCfg
from isaaclab.devices.xrobotoolkit import (
    XRControllerFullBodyDeviceCfg,
    XRTwist2G1RetargeterCfg,
    TwistOutputFormat,
)

# G1 body joint names (29 joints) - order matching TWIST2 policy
G1_BODY_JOINT_NAMES = [
    # Left leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Torso (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


# Default standing pose for reset
G1_TWIST2_DEFAULT_POSE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.793),
    joint_pos={
        # Legs
        "left_hip_pitch_joint": -0.2,
        "left_knee_joint": 0.4,
        "left_ankle_pitch_joint": -0.2,
        "right_hip_pitch_joint": -0.2,
        "right_knee_joint": 0.4,
        "right_ankle_pitch_joint": -0.2,
        # Arms
        "left_shoulder_roll_joint": 0.4,
        "left_elbow_joint": 1.2,
        "right_shoulder_roll_joint": -0.4,
        "right_elbow_joint": 1.2,
        # Others default to 0
        ".*": 0.0,
    },
    joint_vel={".*": 0.0},
)


def reset_to_default_pose(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """Reset robot joints to default standing pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.clone()

    for joint_name, target_pos in G1_TWIST2_DEFAULT_POSE.joint_pos.items():
        if joint_name == ".*":
            continue
        try:
            idx = asset.find_joints(joint_name, preserve_order=True)[0]
            joint_pos[env_ids, idx] = target_pos
        except Exception:
            pass

    # Apply the reset
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(asset.data.joint_vel), env_ids=env_ids)


@configclass
class G1Twist2SceneCfg(InteractiveSceneCfg):
    """Scene configuration for G1 TWIST2 teleoperation."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # G1 robot (floating base for locomotion)
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class G1Twist2ActionsCfg:
    """Action configuration for G1 TWIST2 teleoperation.

    Total action space: 29 DOFs (body joints only)
    """

    body_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_BODY_JOINT_NAMES,
        scale=1.0,
        preserve_order=True,
        use_default_offset=False,  # TWIST2 outputs absolute positions
    )


@configclass
class G1Twist2ObservationsCfg:
    """Observation configuration for G1 TWIST2 teleoperation."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # Joint positions
        joint_pos = ObsTerm(func=mdp.joint_pos)

        # Joint velocities
        joint_vel = ObsTerm(func=mdp.joint_vel)

        # Base orientation (for monitoring)
        base_quat = ObsTerm(func=mdp.root_quat_w)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1Twist2EventsCfg:
    """Event configuration for G1 TWIST2 teleoperation."""

    # Reset robot base position
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),  # Keep at standing height
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # Reset robot joints to default pose
    reset_robot_joints = EventTerm(
        func=reset_to_default_pose,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class G1Twist2TerminationsCfg:
    """Termination configuration for G1 TWIST2 teleoperation."""

    # Time out (very long for teleoperation)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Bad orientation (fallen over)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0},  # ~57 degrees
    )


@configclass
class G1Twist2RewardsCfg:
    """Reward configuration for G1 TWIST2 teleoperation.

    Minimal rewards for teleoperation - mainly for monitoring.
    """

    # Stay alive reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class G1Twist2TeleopEnvCfg(ManagerBasedRLEnvCfg):
    """G1 TWIST2 teleoperation environment configuration.

    This environment enables whole-body teleoperation of the G1 humanoid using
    XR controllers and the TWIST2 imitation learning policy.
    """

    # Scene configuration
    scene: G1Twist2SceneCfg = G1Twist2SceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)

    # Observations
    observations: G1Twist2ObservationsCfg = G1Twist2ObservationsCfg()

    # Actions
    actions: G1Twist2ActionsCfg = G1Twist2ActionsCfg()

    # Events
    events: G1Twist2EventsCfg = G1Twist2EventsCfg()

    # Terminations
    terminations: G1Twist2TerminationsCfg = G1Twist2TerminationsCfg()

    # Rewards
    rewards: G1Twist2RewardsCfg = G1Twist2RewardsCfg()

    def __post_init__(self):
        """Post-initialization configuration."""
        super().__post_init__()

        # Simulation settings
        self.sim.dt = 0.002  # 500 Hz physics
        self.decimation = 10  # 50 Hz control
        self.sim.render_interval = 8

        # Long episode for teleoperation
        self.episode_length_s = 100000.0

        # PhysX settings
        self.sim.physx.solver_type = 1

        # Rendering settings
        self.sim.render = sim_utils.RenderCfg(
            enable_dlssg=True,
            dlss_mode=0,
            rendering_mode="balanced",
        )

        # Teleoperation device configuration
        self.teleop_devices = DevicesCfg(
            devices={
                "xr_controller_full_body": XRControllerFullBodyDeviceCfg(
                    retargeters=[
                        # TWIST2 body retargeter (29 DOFs)
                        XRTwist2G1RetargeterCfg(
                            sim_device=self.sim.device,
                            robot_type="unitree_g1",
                            human_height=None,  # Auto-estimate
                            use_ground_alignment=True,
                            gmr_headless=True,
                            action_scale=0.5,
                            history_length=10,
                            use_threading=True,
                            thread_rate_hz=50.0,
                            output_format=TwistOutputFormat.ABSOLUTE,
                        ),
                    ],
                ),
            }
        )
