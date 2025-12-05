# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T1 humanoid box lifting environment with XR teleoperation."""

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, ArticulationCfg, AssetBase, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

from ..t1_common.joint_names import T1_UPPER_BODY_JOINTS, T1_UPPER_BODY_WITH_GRIPPERS
from ..t1_common.physics_constants import MANIPULATION_PHYSX_SETTINGS
from ..t1_common.t1_camera_cfg import get_default_t1_head_cameras, get_default_t1_wrist_cameras
from ..t1_common.xr_controller_cfg import create_t1_xr_controller_cfg, create_t1_xr_controller_full_body_cfg
from . import t1_lift_box_mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.booster import T1_GRASP_CFG  # isort: skip

# Path to the table and box USD file
TABLE_BOX_USD_PATH = "source/isaaclab_assets/isaaclab_assets/robots/USD/props/Table_and_box_version_test2.usd"


##
# Scene definition
##
@configclass
class T1LiftBoxSceneCfg(InteractiveSceneCfg):
    """Configuration for the T1 box lifting scene."""

    # Robot
    robot: ArticulationCfg = MISSING

    # Table and box (spawned from single USD, box and table are children)
    table_and_box: AssetBaseCfg = MISSING

    # Box object (child of table_and_box USD)
    box: RigidObjectCfg = MISSING

    # Table object (child of table_and_box USD)
    table: RigidObjectCfg = MISSING

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
PREP_STATE = ArticulationCfg.InitialStateCfg(
    pos=(-0.1, 0.0, 0.7),
    rot=(1, 0.0, 0.0, 0),
    joint_pos={
        "AAHead_yaw": 0.0,
        "Head_pitch": 0.0,
        ".*_Shoulder_Pitch": 0.0706,
        "Left_Shoulder_Roll": -1.35,
        "Right_Shoulder_Roll": 1.35,
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
    """Reset robot joints to preparation state and position."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Reset joint positions
    joint_pos = asset.data.default_joint_pos.clone()
    for joint_name, target_pos in PREP_STATE.joint_pos.items():
        idx = asset.find_joints(joint_name, preserve_order=True)[0]
        # find_joints returns a list of indices (even for single matches)
        # We need to handle cases where the pattern matches multiple joints
        if len(idx) == 1:
            # Single joint - simple indexing
            joint_pos[env_ids, idx[0]] = target_pos
        else:
            # Multiple joints matched by regex pattern - broadcast to all matched joints
            joint_pos[env_ids[:, None], idx] = target_pos

    # Reset robot root position and orientation
    root_state = asset.data.default_root_state.clone()
    # Get environment origins for proper multi-environment positioning
    env_origins = env.scene.env_origins[env_ids]
    # Add PREP_STATE position relative to each environment origin
    prep_pos = torch.tensor(PREP_STATE.pos, device=asset.device, dtype=torch.float32)
    root_state[env_ids, :3] = env_origins + prep_pos
    root_state[env_ids, 3:7] = torch.tensor(PREP_STATE.rot, device=asset.device)

    # Apply the reset (pass only the states for env_ids, not the full tensor)
    asset.write_root_state_to_sim(root_state[env_ids], env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos[env_ids], torch.zeros_like(asset.data.joint_vel[env_ids]), env_ids=env_ids)


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=T1_UPPER_BODY_JOINTS,
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
                    joint_names=T1_UPPER_BODY_WITH_GRIPPERS,
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
                    joint_names=T1_UPPER_BODY_WITH_GRIPPERS,
                    preserve_order=True,
                )
            },
        )
        actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "upper_joint_pos"}
        )

        # Object observations
        box_position = ObsTerm(func=t1_lift_box_mdp.box_position_in_robot_frame)
        box_orientation = ObsTerm(func=t1_lift_box_mdp.box_orientation_in_robot_frame)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

            # Add camera observations only if cameras are enabled
            try:
                import carb
                carb_settings = carb.settings.get_settings()
                offscreen_render = bool(carb_settings.get("/isaaclab/render/offscreen"))
                window_enabled = bool(carb_settings.get("/app/window/enabled"))
                cameras_enabled = offscreen_render or window_enabled

                if cameras_enabled:
                    self.head_rgb_cam = ObsTerm(
                        func=mdp.image,
                        params={"sensor_cfg": SceneEntityCfg("head_rgb_cam"), "data_type": "rgb", "normalize": False}
                    )
                    self.left_wrist_cam = ObsTerm(
                        func=mdp.image,
                        params={"sensor_cfg": SceneEntityCfg("left_wrist_cam"), "data_type": "rgb", "normalize": False}
                    )
                    self.right_wrist_cam = ObsTerm(
                        func=mdp.image,
                        params={"sensor_cfg": SceneEntityCfg("right_wrist_cam"), "data_type": "rgb", "normalize": False}
                    )
            except Exception:
                pass

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=reset_to_prep,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    randomize_robot_joint_state = EventTerm(
        func=t1_lift_box_mdp.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.01,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_box_pose = EventTerm(
        func=t1_lift_box_mdp.randomize_box_pose,
        mode="reset",
        params={
            "box_cfg": SceneEntityCfg("box"),
            "table_cfg": SceneEntityCfg("table"),
            "base_pos_xy": (0.5, 0.0),
            "yaw_range_deg": 18.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    box_dropped = DoneTerm(
        func=t1_lift_box_mdp.box_dropped,
        params={"box_cfg": SceneEntityCfg("box"), "height_threshold": 0.2}
    )

    success = DoneTerm(
        func=t1_lift_box_mdp.box_lifted_success,
        params={
            "box_cfg": SceneEntityCfg("box"),
            "height_threshold": 0.75,
        }
    )


@configclass
class RewardCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)


##
# Environment configuration
##
@configclass
class T1LiftBoxEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the T1 box lifting environment."""

    # Scene settings
    scene: T1LiftBoxSceneCfg = T1LiftBoxSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

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
            dlss_mode=2,
            rendering_mode="balanced"
        )

        # Physics settings for manipulation
        for key, value in MANIPULATION_PHYSX_SETTINGS.items():
            setattr(self.sim.physx, key, value)

        # Set T1 robot
        self.scene.robot = T1_GRASP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Check if cameras should be enabled
        self._cameras_enabled = self._check_cameras_enabled()

        # Add head cameras only if enabled (RealSense D455) with 424x240 resolution
        if self._cameras_enabled:
            cameras = get_default_t1_head_cameras(resolution=(240, 424), include_depth=False, include_stereo=False)
            self.scene.head_rgb_cam = cameras["head_rgb_cam"]

            # Add wrist cameras (RealSense D405) with 424x240 resolution
            wrist_cameras = get_default_t1_wrist_cameras(resolution=(240, 424))
            self.scene.left_wrist_cam = wrist_cameras["left_wrist_cam"]
            self.scene.right_wrist_cam = wrist_cameras["right_wrist_cam"]

        # Add table and box from USD
        self.scene.table_and_box = AssetBaseCfg(
            class_type=AssetBase,
            prim_path="{ENV_REGEX_NS}/TableAndBox",
            spawn=UsdFileCfg(usd_path=TABLE_BOX_USD_PATH),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.52, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Box as child of the parent USD
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TableAndBox/box",
            spawn=None,  # already spawned with parent USD
        )

        # Table as child of the parent USD
        self.scene.table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TableAndBox/table",
            spawn=None,  # already spawned with parent USD
        )

        # Configure XR teleoperation
        self.teleop_devices = DevicesCfg(
            devices={
                "xr_controller": create_t1_xr_controller_cfg(sim_device=self.sim.device),
                "xr_controller_full_body": create_t1_xr_controller_full_body_cfg(sim_device=self.sim.device),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )

        # Camera rendering settings for observation recording (only if cameras are enabled)
        if self._cameras_enabled:
            self.rerender_on_reset = True
            self.image_obs_list = ["head_rgb_cam", "left_wrist_cam", "right_wrist_cam"]

    def _check_cameras_enabled(self) -> bool:
        """Check if cameras are enabled via carb settings.

        Cameras are enabled in two scenarios:
        1. When --enable_cameras flag is provided (sets /isaaclab/render/offscreen to True)
        2. When running with GUI (window is enabled, cameras work automatically)

        Returns:
            bool: True if cameras are enabled, False otherwise.
        """
        try:
            import carb
            carb_settings = carb.settings.get_settings()
            offscreen_render = bool(carb_settings.get("/isaaclab/render/offscreen"))
            window_enabled = bool(carb_settings.get("/app/window/enabled"))
            return offscreen_render or window_enabled
        except Exception:
            return False
