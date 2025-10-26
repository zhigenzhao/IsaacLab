# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T1 humanoid cube stacking environment with XR teleoperation."""

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

from ..t1_common.joint_names import T1_UPPER_BODY_JOINTS, T1_UPPER_BODY_WITH_GRIPPERS
from ..t1_common.physics_constants import MANIPULATION_OBJECT_PROPERTIES, MANIPULATION_PHYSX_SETTINGS
from ..t1_common.t1_camera_cfg import get_default_t1_head_cameras
from ..t1_common.xr_controller_cfg import create_t1_xr_controller_cfg
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
PREP_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.1, 0.0, 1.2),
    rot=(1, 0.0, 0.0, 0),
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
    """Reset robot joints to preparation state and position."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Reset joint positions
    joint_pos = asset.data.default_joint_pos.clone()
    for joint_name, target_pos in PREP_STATE.joint_pos.items():
        idx = asset.find_joints(joint_name, preserve_order=True)[0]
        joint_pos[env_ids, idx] = target_pos

    # Reset robot root position and orientation
    root_state = asset.data.default_root_state.clone()
    root_state[env_ids, :3] = torch.tensor(PREP_STATE.pos, device=asset.device)
    root_state[env_ids, 3:7] = torch.tensor(PREP_STATE.rot, device=asset.device)

    # Apply the reset
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(asset.data.joint_vel), env_ids=env_ids)


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
        cube_positions = ObsTerm(func=t1_stack_mdp.cube_positions_in_robot_frame)
        cube_orientations = ObsTerm(func=t1_stack_mdp.cube_orientations_in_robot_frame)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

            # Add camera observations only if cameras are enabled
            # This is determined at environment creation time based on --enable_cameras flag
            try:
                import carb
                carb_settings = carb.settings.get_settings()
                # Check multiple settings to determine if cameras are enabled:
                # - /isaaclab/render/offscreen: True in headless mode with cameras
                # - /app/window/enabled: False in headless mode
                # If window is enabled (GUI mode), we still want cameras if they exist in the scene
                offscreen_render = bool(carb_settings.get("/isaaclab/render/offscreen"))
                window_enabled = bool(carb_settings.get("/app/window/enabled"))

                # Cameras are enabled if:
                # 1. Offscreen rendering is enabled (headless + --enable_cameras), OR
                # 2. Window is enabled (GUI mode, cameras always work)
                cameras_enabled = offscreen_render or window_enabled

                if cameras_enabled:
                    # Add camera observation terms dynamically
                    self.head_rgb_cam = ObsTerm(
                        func=mdp.image,
                        params={"sensor_cfg": SceneEntityCfg("head_rgb_cam"), "data_type": "rgb", "normalize": False}
                    )
                    self.head_depth_cam = ObsTerm(
                        func=mdp.image,
                        params={"sensor_cfg": SceneEntityCfg("head_depth_cam"), "data_type": "distance_to_image_plane", "normalize": False}
                    )
            except Exception:
                # If carb settings are not available, cameras are not enabled
                pass

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
            enable_dlssg=False,
            dlss_mode=0,
            rendering_mode="balanced"
        )

        # Physics settings for stacking
        for key, value in MANIPULATION_PHYSX_SETTINGS.items():
            setattr(self.sim.physx, key, value)

        # Set T1 robot
        self.scene.robot = T1_GRASP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Check if cameras should be enabled (only when --enable_cameras flag is set)
        # This improves performance when cameras are not needed
        self._cameras_enabled = self._check_cameras_enabled()

        # Add head cameras only if enabled (RealSense D455) with 424x240 resolution
        if self._cameras_enabled:
            cameras = get_default_t1_head_cameras(resolution=(240, 424), include_depth=True, include_stereo=False)
            self.scene.head_rgb_cam = cameras["head_rgb_cam"]
            self.scene.head_depth_cam = cameras["head_depth_cam"]

        # Add cubes to scene (on table surface at z=1.0703)
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 1.0703], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=MANIPULATION_OBJECT_PROPERTIES,
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 1.0703], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=MANIPULATION_OBJECT_PROPERTIES,
            ),
        )
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 1.0703], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=MANIPULATION_OBJECT_PROPERTIES,
            ),
        )

        # No FrameTransformer needed - we'll get hand poses directly from articulation body data

        # Configure XR teleoperation
        self.teleop_devices = DevicesCfg(
            devices={
                "xr_controller": create_t1_xr_controller_cfg(sim_device=self.sim.device),
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
            self.image_obs_list = ["head_rgb_cam", "head_depth_cam"]

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
            # Check if offscreen rendering is enabled (headless + --enable_cameras)
            offscreen_render = bool(carb_settings.get("/isaaclab/render/offscreen"))
            # Check if window is enabled (GUI mode)
            window_enabled = bool(carb_settings.get("/app/window/enabled"))

            # Cameras are enabled if either offscreen rendering or GUI window is active
            return offscreen_render or window_enabled
        except Exception:
            # If carb settings are not available yet, default to False
            return False
