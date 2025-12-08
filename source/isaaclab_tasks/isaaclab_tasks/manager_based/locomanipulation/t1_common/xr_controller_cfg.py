# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common XR controller configuration for T1 environments."""

from isaaclab.devices.xrobotoolkit import XRControllerDeviceCfg, XRControllerFullBodyDeviceCfg
from isaaclab.devices.xrobotoolkit.retargeters import (
    TwistOutputFormat,
    XRGripperRetargeterCfg,
    XRT1GMRRetargeterCfg,
    XRT1MinkIKRetargeterCfg,
    XRTwistRetargeterCfg,
)


def create_t1_xr_controller_cfg(
    sim_device: str,
    ik_rate_hz: float = 90.0,
    collision_avoidance_distance: float = 0.04,
    collision_detection_distance: float = 0.10,
    velocity_limit_factor: float = 0.7,
    reference_frame: str = "trunk",
    enable_head_tracking: bool = False,
) -> XRControllerDeviceCfg:
    """Create standard T1 XR controller configuration.

    Args:
        sim_device: Simulation device (e.g., "cuda:0")
        ik_rate_hz: IK solver update rate
        collision_avoidance_distance: Distance for collision avoidance
        collision_detection_distance: Distance for collision detection
        velocity_limit_factor: Velocity limit factor for IK
        reference_frame: Reference frame for relative control
        enable_head_tracking: Enable head tracking with HMD orientation (via direct joint control)

    Returns:
        Configured XR controller device
    """
    return XRControllerDeviceCfg(
        control_mode="dual_hand",
        gripper_source="trigger",
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
        deadzone_threshold=0.01,
        retargeters=[
            XRT1MinkIKRetargeterCfg(
                xml_path="source/isaaclab_assets/isaaclab_assets/robots/xmls/scene_t1_ik.xml",
                headless=True,
                ik_rate_hz=ik_rate_hz,
                collision_avoidance_distance=collision_avoidance_distance,
                collision_detection_distance=collision_detection_distance,
                velocity_limit_factor=velocity_limit_factor,
                output_joint_positions_only=True,
                sim_device=sim_device,
                reference_frame=reference_frame,
                enable_head_tracking=enable_head_tracking,
                # Motion tracker configuration for elbow tracking
                motion_tracker_config={
                    "right_arm": {
                        "serial": "PC2310MLK6140013G",
                        "link_target": "right_elbow"
                    },
                    "left_arm": {
                        "serial": "PC2310MLK6140583G",
                        "link_target": "left_elbow"
                    }
                },
                arm_length_scale_factor=1.0,
                motion_tracker_task_weight=0.8,
            ),
            XRGripperRetargeterCfg(
                control_hand="left",
                input_source="trigger",
                mode="binary",
                binary_threshold=0.5,
                invert=False,
                open_value=-0.523,
                closed_value=1.57,
                sim_device=sim_device,
            ),
            XRGripperRetargeterCfg(
                control_hand="right",
                input_source="trigger",
                mode="binary",
                binary_threshold=0.5,
                invert=False,
                open_value=-0.523,
                closed_value=1.57,
                sim_device=sim_device,
            ),
        ],
        sim_device=sim_device,
    )


def create_t1_xr_controller_full_body_cfg(
    sim_device: str,
    human_height: float | None = None,
    use_ground_alignment: bool = True,
    ground_offset: float = 0.0,
    headless: bool = False,
    show_human_skeleton: bool = True,
    viewer_fps: int = 30,
) -> XRControllerFullBodyDeviceCfg:
    """Create T1 XR full-body controller configuration with GMR retargeting.

    This configuration uses XRControllerFullBodyDevice with 24-joint body tracking
    and XRT1GMRRetargeter for whole-body motion retargeting via GMR (General Motion Retargeting).

    Args:
        sim_device: Simulation device (e.g., "cuda:0")
        human_height: Human height in meters. If None, auto-estimate from first frame.
        use_ground_alignment: If True, automatically align body to ground plane (important for headset-relative tracking).
        ground_offset: Manual ground offset adjustment in meters.
        headless: If True, run without MuJoCo viewer visualization. If False, display retargeting result in viewer.
        show_human_skeleton: If True (and headless=False), display human skeleton visualization alongside robot.
        viewer_fps: Frames per second for MuJoCo viewer update (when headless=False).

    Returns:
        Configured XR full-body controller device with GMR retargeting

    Note:
        - XRT1GMRRetargeter outputs 16 upper body joints (2 head + 14 arms) to match XRT1MinkIKRetargeter format
        - Head joints default to [0.0, 0.0] as GMR's T1 model doesn't include head tracking
        - For head tracking, use XRT1MinkIKRetargeter with XRControllerDevice instead
    """
    return XRControllerFullBodyDeviceCfg(
        retargeters=[
            XRT1GMRRetargeterCfg(
                human_height=human_height,
                use_ground_alignment=use_ground_alignment,
                ground_offset=ground_offset,
                headless=headless,
                show_human_skeleton=show_human_skeleton,
                viewer_fps=viewer_fps,
                sim_device=sim_device,
            ),
            XRGripperRetargeterCfg(
                control_hand="left",
                input_source="trigger",
                mode="binary",
                binary_threshold=0.5,
                invert=False,
                open_value=-0.523,
                closed_value=1.57,
                sim_device=sim_device,
            ),
            XRGripperRetargeterCfg(
                control_hand="right",
                input_source="trigger",
                mode="binary",
                binary_threshold=0.5,
                invert=False,
                open_value=-0.523,
                closed_value=1.57,
                sim_device=sim_device,
            ),
        ],
        sim_device=sim_device,
    )


def create_t1_xr_twist_cfg(
    twist_config_path: str,
    sim_device: str = "cuda:0",
    human_height: float | None = None,
    use_threading: bool = True,
    thread_rate_hz: float = 50.0,
    enable_grippers: bool = True,
    gripper_open_value: float = -0.523,
    gripper_closed_value: float = 1.57,
) -> XRControllerFullBodyDeviceCfg:
    """Create XRControllerFullBodyDevice config with TWIST retargeter for T1.

    This configuration uses XRControllerFullBodyDevice with 24-joint body tracking
    and XRTwistRetargeter for whole-body motion tracking via the TWIST policy.

    Args:
        twist_config_path: Path to TWIST configuration YAML file containing model path,
            joint indices, and policy parameters.
        sim_device: Simulation device (e.g., "cuda:0")
        human_height: Human height in meters. If None, auto-estimate from first frame.
        use_threading: If True, run TWIST inference in a separate thread.
        thread_rate_hz: Thread update rate in Hz (when use_threading=True).
        enable_grippers: If True, include gripper retargeters for binary gripper control.
        gripper_open_value: Joint position for open gripper state.
        gripper_closed_value: Joint position for closed gripper state.

    Returns:
        Configured XR full-body controller device with TWIST retargeting

    Note:
        The TWIST config YAML must contain:
        - model_path: Path to ONNX model
        - TWIST_POLICY_DOFS: Number of policy DOFs (e.g., 27)
        - TWIST_HIST_LEN: Observation history length (e.g., 10)
        - TWIST_POLICY_INDICES: Joint indices for policy output
        - TWIST_DEFAULT_JOINT_POS: Default joint positions
        - POLICY_ACTION_SCALE: Action scaling factor
    """
    retargeters = [
        XRTwistRetargeterCfg(
            twist_config_path=twist_config_path,
            robot_type="booster_t1_29dof",
            human_height=human_height,
            use_ground_alignment=True,
            use_threading=use_threading,
            thread_rate_hz=thread_rate_hz,
            output_format=TwistOutputFormat.ABSOLUTE,
            sim_device=sim_device,
        ),
    ]

    # Add gripper retargeters for binary gripper control via XR triggers
    if enable_grippers:
        retargeters.extend([
            XRGripperRetargeterCfg(
                control_hand="left",
                input_source="trigger",
                mode="binary",
                binary_threshold=0.5,
                invert=False,
                open_value=gripper_open_value,
                closed_value=gripper_closed_value,
                sim_device=sim_device,
            ),
            XRGripperRetargeterCfg(
                control_hand="right",
                input_source="trigger",
                mode="binary",
                binary_threshold=0.5,
                invert=False,
                open_value=gripper_open_value,
                closed_value=gripper_closed_value,
                sim_device=sim_device,
            ),
        ])

    return XRControllerFullBodyDeviceCfg(
        retargeters=retargeters,
        sim_device=sim_device,
    )
