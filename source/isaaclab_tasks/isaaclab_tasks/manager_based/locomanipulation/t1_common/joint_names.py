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
