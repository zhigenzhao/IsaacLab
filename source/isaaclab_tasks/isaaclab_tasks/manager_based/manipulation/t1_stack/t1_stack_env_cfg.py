# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T1 humanoid cube stacking environment with XR teleoperation."""

import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, Articulation
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.xrobotoolkit import (
    XRControllerDeviceCfg,
    XRT1MinkIKRetargeterCfg,
    XRGripperRetargeterCfg,
)

import isaaclab.envs.mdp as mdp
from . import t1_stack_mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.booster import T1_GRASP_CFG  # isort: skip


##
# Scene definition
##
@configclass
class T1StackSceneCfg(InteractiveSceneCfg):
    """Configuration for the T1 stacking scene."""

    # Robot
    robot: ArticulationCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 1.05], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# Reset state for T1
##
PREP_STATE = {
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    ".*_Shoulder_Pitch": 0.0706,
    "Left_Shoulder_Roll": -1.55,
    "Right_Shoulder_Roll": 1.55,
    ".*_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": 0.0,
    "Right_Elbow_Yaw": 0.0,
    ".*_Wrist_Pitch": 0.0,
    ".*_Wrist_Yaw": 0.0,
    ".*_Hand_Roll": 0.0,
    "Waist": 0.0,
}


def reset_to_prep(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """Reset robot joints to preparation state."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.clone()
    for joint_name, target_pos in PREP_STATE.items():
        idx = asset.find_joints(joint_name, preserve_order=True)[0]
        joint_pos[env_ids, idx] = target_pos

    # Apply the reset
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(asset.data.joint_vel), env_ids=env_ids)


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

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
        use_default_offset=False
    )
    left_gripper = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_Link1"],
        scale=1.0,
        preserve_order=True,
        use_default_offset=False
    )
    right_gripper = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_Link1"],
        scale=1.0,
        preserve_order=True,
        use_default_offset=False
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot state
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
                        "left_Link1",
                        "left_Link2",
                        "right_Link1",
                        "right_Link2",
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
                        "left_Link1",
                        "left_Link2",
                        "right_Link1",
                        "right_Link2",
                    ],
                    preserve_order=True,
                )
            },
        )
        actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "upper_joint_pos"}
        )

        # Object observations
        cube_positions = ObsTerm(func=t1_stack_mdp.cube_positions_in_robot_frame)
        cube_orientations = ObsTerm(func=t1_stack_mdp.cube_orientations_in_robot_frame)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        # Left hand grasping
        left_grasp_1 = ObsTerm(
            func=t1_stack_mdp.object_grasped_by_hand,
            params={
                "hand": "left",
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        left_grasp_2 = ObsTerm(
            func=t1_stack_mdp.object_grasped_by_hand,
            params={
                "hand": "left",
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        # Right hand grasping
        right_grasp_1 = ObsTerm(
            func=t1_stack_mdp.object_grasped_by_hand,
            params={
                "hand": "right",
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        right_grasp_2 = ObsTerm(
            func=t1_stack_mdp.object_grasped_by_hand,
            params={
                "hand": "right",
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        # Stacking checks
        stack_1 = ObsTerm(
            func=t1_stack_mdp.object_stacked,
            params={
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=reset_to_prep,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    randomize_robot_joint_state = EventTerm(
        func=t1_stack_mdp.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.01,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cube_positions = EventTerm(
        func=t1_stack_mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.15, 0.15), "z": (1.0703, 1.0703), "yaw": (-1.0, 1.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")}
    )

    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")}
    )

    cube_3_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_3")}
    )

    success = DoneTerm(func=t1_stack_mdp.cubes_stacked)


@configclass
class RewardCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)


##
# Environment configuration
##
@configclass
class T1CubeStackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the T1 cube stacking environment."""

    # Scene settings
    scene: T1StackSceneCfg = T1StackSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    # MDP settings
    rewards: RewardCfg = RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 10
        self.episode_length_s = 30.0

        # Simulation settings
        self.sim.dt = 0.002  # 500Hz physics
        self.sim.render_interval = 8
        self.sim.physx.solver_type = 1

        # Enable DLSS rendering
        self.sim.render = sim_utils.RenderCfg(
            enable_dlssg=True,
            dlss_mode=0,
            rendering_mode="balanced"
        )

        # Physics settings for stacking
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Set T1 robot
        self.scene.robot = T1_GRASP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Cube properties
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Add cubes to scene (on table surface at z=1.0703)
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 1.0703], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 1.0703], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
            ),
        )
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 1.0703], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
            ),
        )

        # No FrameTransformer needed - we'll get hand poses directly from articulation body data

        # Configure XR teleoperation
        self.teleop_devices = DevicesCfg(
            devices={
                "xr_controller": XRControllerDeviceCfg(
                    control_mode="dual_hand",
                    gripper_source="trigger",
                    pos_sensitivity=1.0,
                    rot_sensitivity=1.0,
                    deadzone_threshold=0.01,

                    retargeters=[
                        XRT1MinkIKRetargeterCfg(
                            xml_path="source/isaaclab_assets/isaaclab_assets/robots/xmls/scene_t1_ik.xml",
                            headless=False,
                            ik_rate_hz=90.0,
                            collision_avoidance_distance=0.04,
                            collision_detection_distance=0.10,
                            velocity_limit_factor=0.7,
                            output_joint_positions_only=True,
                            sim_device=self.sim.device,
                        ),
                        XRGripperRetargeterCfg(
                            control_hand="left",
                            input_source="trigger",
                            mode="binary",
                            binary_threshold=0.5,
                            invert=True,
                            open_value=-0.523,
                            closed_value=1.57,
                            sim_device=self.sim.device,
                        ),
                        XRGripperRetargeterCfg(
                            control_hand="right",
                            input_source="trigger",
                            mode="binary",
                            binary_threshold=0.5,
                            invert=True,
                            open_value=-0.523,
                            closed_value=1.57,
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )
