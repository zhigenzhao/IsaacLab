# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common joint name constants for T1 robot environments."""

# Upper body joints (16 joints: head + arms)
T1_UPPER_BODY_JOINTS = [
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
]

# Upper body joints with waist and grippers (20 joints)
T1_UPPER_BODY_WITH_GRIPPERS = [
    *T1_UPPER_BODY_JOINTS,
    "Waist",
    "left_Link1",
    "left_Link2",
    "right_Link1",
    "right_Link2",
]

# Full body joints (29 joints: head + arms + torso + legs)
# Order matches TWIST policy output
T1_FULL_BODY_JOINTS = [
    # Head (2)
    "AAHead_yaw",
    "Head_pitch",
    # Left arm (7)
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Left_Wrist_Pitch",
    "Left_Wrist_Yaw",
    "Left_Hand_Roll",
    # Right arm (7)
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Right_Wrist_Pitch",
    "Right_Wrist_Yaw",
    "Right_Hand_Roll",
    # Torso (1)
    "Waist",
    # Left leg (6)
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    # Right leg (6)
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

# Gripper joints (main actuated joints only, mimic joints follow automatically)
T1_GRIPPER_JOINTS = [
    "left_Link1",
    "right_Link1",
]

# Gripper mimic joints (follow main gripper joints)
T1_GRIPPER_MIMIC_JOINTS = [
    "left_Link11",
    "left_Link2",
    "left_Link22",
    "right_Link11",
    "right_Link2",
    "right_Link22",
]
