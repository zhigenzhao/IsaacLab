# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnvCfg, ManagerBasedRLEnv
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab_assets.robots.booster import T1_CFG, T1_REACH_CFG  # isort: skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab.envs.mdp as mdp
from .t1_teleop_commands_cfg import DummyCommandCfg
from .t1_teleop_observations import ObservationCfg
from .t1_teleop_recorders import PostStepCommandRecorderCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import RecorderTermCfg


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

    # robots
    robot: ArticulationCfg = MISSING


PREP_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.68),  # 0.72
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
        ".*_Hip_Pitch": -0.2,
        ".*_Hip_Roll": 0.0,
        ".*_Hip_Yaw": 0.0,
        ".*_Knee_Pitch": 0.4,
        ".*_Ankle_Pitch": -0.25,
        ".*_Ankle_Roll": 0.0,
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
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.64, 0.68),
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (-0.4, 0.4),
                "y": (-0.4, 0.4),
                "z": (-0.4, 0.4),
                "roll": (-0.4, 0.4),
                "pitch": (-0.4, 0.4),
                "yaw": (-0.4, 0.4),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=reset_to_prep,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")}
    )


@configclass
class TerminationCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CommandCfg:
    base_velocity = DummyCommandCfg(dim=3)


@configclass
class ActionCfg:
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
        use_default_offset=True
    )

    lower_joint_pos = mdp.JointPositionActionCfg(  # type: ignore
        asset_name="robot",
        joint_names=[
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
        scale={
            "Waist": 0.0,
            "Left_Hip_Pitch": 1.0,
            "Left_Hip_Roll": 1.0,
            "Left_Hip_Yaw": 1.0,
            "Left_Knee_Pitch": 1.0,
            "Left_Ankle_Pitch": 1.0,
            "Left_Ankle_Roll": 1.0,
            "Right_Hip_Pitch": 1.0,
            "Right_Hip_Roll": 1.0,
            "Right_Hip_Yaw": 1.0,
            "Right_Knee_Pitch": 1.0,
            "Right_Ankle_Pitch": 1.0,
            "Right_Ankle_Roll": 1.0,
        },
        preserve_order=True,
        use_default_offset=True,
    )

    # gripper_pose = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         "left_Link11",
    #         "left_Link22",
    #         "right_Link11",
    #         "right_Link22"
    #     ]
    # )


@configclass
class RewardCfg:
    alive = RewTerm(mdp.is_alive, weight=1.0)


@configclass
class RecorderCfg(ActionStateRecorderManagerCfg):
    record_post_step_commands: PostStepCommandRecorderCfg = PostStepCommandRecorderCfg()


@configclass
class T1TeleopEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationCfg = ObservationCfg()
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
        self.scene.robot = T1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        self.sim.physx.solver_type = 1
        import isaaclab.sim as sim_utils
        self.sim.render = sim_utils.RenderCfg(
            enable_dlssg=True,
            dlss_mode=0,  # Set DLSS to Performance Mode
            rendering_mode="balanced"
        )
