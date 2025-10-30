# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T1 Teleoperation Reach Environment Configuration (Upper Body, No Grippers)."""

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_assets.robots.booster import T1_REACH_CFG  # isort: skip

from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab.envs.mdp as mdp
from .t1_teleop_commands_cfg import DummyCommandCfg
from .t1_teleop_recorders import PostStepCommandRecorderCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.xrobotoolkit import (
    XRControllerDeviceCfg,
    XRT1MinkIKRetargeterCfg,
)
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise


@configclass
class SceneCfg(InteractiveSceneCfg):
    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # robot
    robot: ArticulationCfg = T1_REACH_CFG


PREP_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.7),
    joint_pos={
        "AAHead_yaw": 0.0,
        "Head_pitch": 0.0,
        ".*_Shoulder_Pitch": 0.0706,
        "Left_Shoulder_Roll": -1.55,
        "Right_Shoulder_Roll": 1.55,
        ".*_Elbow_Pitch": 0.0,
        "Left_Elbow_Yaw": -1.57,
        "Right_Elbow_Yaw": 1.57,
        ".*_Wrist_Pitch": 0.0,
        ".*_Wrist_Yaw": 0.0,
        ".*_Hand_Roll": 0.0,
        "Waist": 0.0,
    },
    joint_vel={".*": 0.0},
)


def reset_to_prep(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """Reset robot joints to preparation state."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.clone()
    for joint_name, target_pos in PREP_STATE.joint_pos.items():
        idx = asset.find_joints(joint_name, preserve_order=True)[0]
        joint_pos[env_ids, idx] = target_pos

    # Apply the reset
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(asset.data.joint_vel), env_ids=env_ids)


@configclass
class EventCfg:
    """Configuration for events."""
    reset_robot_joints = EventTerm(
        func=reset_to_prep,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")}
    )


@configclass
class TerminationCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CommandCfg:
    base_velocity = DummyCommandCfg(dim=3)


@configclass
class ActionCfg:
    """Action configuration for upper body only (no grippers)."""
    upper_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
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
            "Right_Hand_Roll"
        ],
        scale=1.0,
        preserve_order=True,
        use_default_offset=False  # Mink IK outputs absolute positions, no offset needed
    )


@configclass
class RewardCfg:
    alive = RewTerm(mdp.is_alive, weight=1.0)


@configclass
class RecorderCfg(ActionStateRecorderManagerCfg):
    record_post_step_commands: PostStepCommandRecorderCfg = PostStepCommandRecorderCfg()


@configclass
class ReachObservationCfg:
    """Observation specifications for reaching tasks (upper body only, no grippers)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=1,
            noise=Gnoise(mean=0.0, std=0.1),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Gnoise(mean=0.0, std=0.01),
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
                    ],
                    preserve_order=True,
                )
            },
        )
        actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "upper_joint_pos"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class T1TeleopReachEnvCfg(ManagerBasedRLEnvCfg):
    """T1 teleoperation environment configuration for reaching tasks (no gripper control)."""

    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ReachObservationCfg = ReachObservationCfg()
    actions: ActionCfg = ActionCfg()
    terminations: TerminationCfg = TerminationCfg()
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()
    commands: CommandCfg = CommandCfg()
    recorders: RecorderCfg = RecorderCfg()

    def __post_init__(self):
        super().__post_init__()

        # sim settings
        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.render_interval = 8
        self.episode_length_s = 1000000.0
        self.scene.robot = T1_REACH_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        self.sim.physx.solver_type = 1
        self.sim.render = sim_utils.RenderCfg(
            enable_dlssg=True,
            dlss_mode=0,  # Set DLSS to Performance Mode
            rendering_mode="balanced"
        )

        # Configure teleoperation devices (no gripper retargeters)
        self.teleop_devices = DevicesCfg(
            devices={
                # XRoboToolkit VR controller device
                "xr_controller": XRControllerDeviceCfg(
                    # Controller settings
                    control_mode="dual_hand",      # Use both controllers for T1 upper body
                    gripper_source="trigger",       # Not used but required field
                    pos_sensitivity=1.0,            # Controller position sensitivity
                    rot_sensitivity=1.0,            # Controller rotation sensitivity
                    deadzone_threshold=0.01,        # Minimum movement threshold

                    # Mink IK retargeter for T1 upper body control (no gripper control)
                    retargeters=[
                        XRT1MinkIKRetargeterCfg(
                            xml_path="source/isaaclab_assets/isaaclab_assets/robots/xmls/scene_t1_ik.xml",
                            headless=False,
                            ik_rate_hz=100.0,
                            collision_avoidance_distance=0.04,
                            collision_detection_distance=0.10,
                            velocity_limit_factor=0.7,
                            output_joint_positions_only=True,  # Output only 16 joint positions
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                ),
                # Keyboard as fallback
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )
