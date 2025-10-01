# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Booster Robotics robots.

The following configurations are available:

* :obj:`T1_CFG`: T1 humanoid robot with 29 DOF (7-DOF arms, mobile base)
* :obj:`T1_REACH_CFG`: Fixed base version of T1 with upper body only
* :obj:`T1_MINIMAL_CFG`: T1 with reduced collision meshes for faster simulation

Reference: https://booster.feishu.cn/wiki/UvowwBes1iNvvUkoeeVc3p5wnUg
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##


T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/isaaclab_assets/isaaclab_assets/robots/USD/t1_29dof_no_upper_collision.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": 1.35,
            ".*_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
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
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "head": ImplicitActuatorCfg(
            effort_limit_sim=7,
            velocity_limit_sim=12.56,
            joint_names_expr=["Head_pitch", "AAHead_yaw"],
            stiffness=20.0,
            damping=0.2,
            armature=0.01,
        ),
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Yaw",
                ".*_Hip_Roll",
                ".*_Hip_Pitch",
                ".*_Knee_Pitch",
                "Waist",
            ],
            effort_limit_sim={
                ".*_Hip_Yaw": 60.0,
                ".*_Hip_Roll": 25.0,
                ".*_Hip_Pitch": 30.0,
                ".*_Knee_Pitch": 60.0,
                "Waist": 30.0,
            },
            velocity_limit_sim={
                ".*_Hip_Yaw": 10.9,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Pitch": 12.5,
                ".*_Knee_Pitch": 11.7,
                "Waist": 10.88,
            },
            stiffness=200.0,
            damping=5.0,
            armature={
                ".*_Hip_.*": 0.01,
                ".*_Knee_Pitch": 0.01,
                "Waist": 0.01,
            },
            min_delay=10,
            max_delay=20,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit_sim={
                ".*_Ankle_Pitch": 24.0,
                ".*_Ankle_Roll": 15.0,
            },
            velocity_limit_sim={
                ".*_Ankle_Pitch": 18.8,
                ".*_Ankle_Roll": 12.4,
            },
            stiffness=50.0,
            damping=3.0,
            armature=0.01,
            min_delay=10,
            max_delay=20,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
                ".*_Wrist_Pitch",
                ".*_Wrist_Yaw",
                ".*_Hand_Roll",
            ],
            effort_limit_sim={
                ".*_Shoulder_.*": 10.0,
                ".*_Elbow_.*": 10.0,
                ".*_Wrist_.*": 10.0,
                ".*_Hand_Roll": 10.0,
            },
            velocity_limit_sim={
                ".*_Shoulder_.*": 18.84,
                ".*_Elbow_.*": 18.84,
                ".*_Wrist_.*": 18.84,
                ".*_Hand_Roll": 18.84,
            },
            stiffness=50.0,
            damping=3.0,
            armature={
                ".*_Shoulder_.*": 0.01,
                ".*_Elbow_.*": 0.01,
                ".*_Wrist_.*": 0.001,
                ".*_Hand_Roll": 0.001,
            },
        ),
    },
)
"""Configuration for the Booster T1 humanoid robot with 29 DOF."""


T1_REACH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/isaaclab_assets/isaaclab_assets/robots/USD/T1_7dof_arms_with_gripper_Fixed_upper_only.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": 1.35,
            ".*_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            ".*_Wrist_Pitch": 0.0,
            ".*_Wrist_Yaw": 0.0,
            ".*_Hand_Roll": 0.0,
            "Waist": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "head": ImplicitActuatorCfg(
            effort_limit_sim=7,
            velocity_limit_sim=12.56,
            joint_names_expr=["Head_pitch", "AAHead_yaw"],
            stiffness=20.0,
            damping=0.2,
            armature=0.01,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["Waist"],
            effort_limit_sim=30.0,
            velocity_limit_sim=10.88,
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
                ".*_Wrist_Pitch",
                ".*_Wrist_Yaw",
                ".*_Hand_Roll",
            ],
            effort_limit_sim={
                ".*_Shoulder_.*": 200.0,
                ".*_Elbow_.*": 200.0,
                ".*_Wrist_.*": 200.0,
                ".*_Hand_Roll": 200.0,
            },
            velocity_limit_sim={
                ".*_Shoulder_.*": 18.84,
                ".*_Elbow_.*": 18.84,
                ".*_Wrist_.*": 18.84,
                ".*_Hand_Roll": 18.84,
            },
            stiffness=2000.0,
            damping=3.0,
            armature={
                ".*_Shoulder_.*": 0.01,
                ".*_Elbow_.*": 0.01,
                ".*_Wrist_.*": 0.001,
                ".*_Hand_Roll": 0.001,
            },
        ),
    },
)
"""Configuration for the Booster T1 robot with fixed base (upper body only)."""


T1_MINIMAL_CFG = T1_CFG.copy()
"""Configuration for the Booster T1 humanoid robot with reduced collision meshes for faster simulation."""