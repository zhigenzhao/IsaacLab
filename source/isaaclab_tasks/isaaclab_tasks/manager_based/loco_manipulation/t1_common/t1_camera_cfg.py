# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera configurations for T1 robot with RealSense D455 sensor.

This module provides camera configurations for the RealSense D455 camera mounted
on the T1 robot's H2 head link. The configurations are based on the actual USD
geometry found at /T1/H2/Realsense in the robot model.

RealSense D455 Camera Specifications:
- Color Camera: OmniVision OV9782
- Depth Camera: Pseudo-depth sensor
- Focal Length: 1.93mm
- Horizontal Aperture: 3.896mm
- Vertical Aperture: 2.453mm
- Field of View: ~87° horizontal, ~58° vertical
"""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

##
# Camera Configurations
##


class T1HeadCameraCfg:
    """Camera configurations for the T1 robot's head-mounted RealSense D455.

    The RealSense D455 is mounted on the H2 link of the T1 robot and includes:
    - RGB color camera for visual observations
    - Depth camera for distance measurements
    - Left and Right stereo cameras (optional, for advanced depth perception)

    All cameras share the same intrinsic parameters but have different positions
    relative to the H2 link.
    """

    @staticmethod
    def create_head_rgb_camera(
        prim_path: str = "{ENV_REGEX_NS}/Robot/H2/head_rgb_cam",
        height: int = 480,
        width: int = 640,
        update_period: float = 0.0,
        data_types: list[str] | None = None,
    ) -> CameraCfg:
        """Create RGB color camera configuration for the T1 head.

        This camera is positioned at the RealSense D455's color camera location,
        offset by (0, 0.0115, 0) from the H2 link origin.

        Args:
            prim_path: USD path for the camera prim. Defaults to H2 link.
            height: Image height in pixels. Default 480 (VGA resolution).
            width: Image width in pixels. Default 640 (VGA resolution).
            update_period: Camera update period in seconds. 0.0 = every frame.
            data_types: List of data types to capture. Default ["rgb"].

        Returns:
            CameraCfg: Camera configuration for the head RGB camera.
        """
        if data_types is None:
            data_types = ["rgb"]

        return CameraCfg(
            prim_path=prim_path,
            update_period=update_period,
            height=height,
            width=width,
            data_types=data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.93,
                focus_distance=400.0,
                horizontal_aperture=3.896,
                clipping_range=(0.1, 10.0),  # RealSense D455 range: 0.1m - 10m
            ),
            offset=CameraCfg.OffsetCfg(
                # Color camera is offset by 11.5mm in Y direction from H2 origin
                # Transform accounts for ROS convention (X forward, Y left, Z up)
                pos=(0.0, 0.0115, 0.0),
                # Rotation from USD local transform (camera looks forward)
                # Original: ((-2.2e-16, 1, 2.7e-16, 0), (5e-16, -2.2e-16, 1, 0), (1, 2.7e-16, -4.4e-16, 0))
                # Simplified: 90° rotation around Y, then 90° around Z
                rot=(0.5, -0.5, 0.5, -0.5),  # Camera points forward in ROS convention
                convention="ros",
            ),
        )

    @staticmethod
    def create_head_depth_camera(
        prim_path: str = "{ENV_REGEX_NS}/Robot/H2/head_depth_cam",
        height: int = 480,
        width: int = 640,
        update_period: float = 0.0,
        data_types: list[str] | None = None,
    ) -> CameraCfg:
        """Create depth camera configuration for the T1 head.

        This camera is positioned at the RealSense D455's depth sensor location,
        at the H2 link origin (0, 0, 0).

        Args:
            prim_path: USD path for the camera prim. Defaults to H2 link.
            height: Image height in pixels. Default 480 (VGA resolution).
            width: Image width in pixels. Default 640 (VGA resolution).
            update_period: Camera update period in seconds. 0.0 = every frame.
            data_types: List of data types to capture. Default ["distance_to_image_plane"].

        Returns:
            CameraCfg: Camera configuration for the head depth camera.
        """
        if data_types is None:
            data_types = ["distance_to_image_plane"]

        return CameraCfg(
            prim_path=prim_path,
            update_period=update_period,
            height=height,
            width=width,
            data_types=data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.93,
                focus_distance=400.0,
                horizontal_aperture=3.896,
                clipping_range=(0.1, 10.0),  # RealSense D455 depth range
            ),
            offset=CameraCfg.OffsetCfg(
                # Depth camera is at H2 origin
                pos=(0.0, 0.0, 0.0),
                # Rotation for depth camera pointing forward
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )

    @staticmethod
    def create_head_stereo_left_camera(
        prim_path: str = "{ENV_REGEX_NS}/Robot/H2/head_stereo_left_cam",
        height: int = 480,
        width: int = 640,
        update_period: float = 0.0,
        data_types: list[str] | None = None,
    ) -> CameraCfg:
        """Create left stereo camera configuration for the T1 head.

        This camera is positioned at the RealSense D455's left stereo camera location,
        offset by (0, -0.0475, 0) from the H2 link origin (47.5mm baseline).

        Args:
            prim_path: USD path for the camera prim.
            height: Image height in pixels. Default 480 (VGA resolution).
            width: Image width in pixels. Default 640 (VGA resolution).
            update_period: Camera update period in seconds. 0.0 = every frame.
            data_types: List of data types to capture. Default ["rgb"].

        Returns:
            CameraCfg: Camera configuration for the left stereo camera.
        """
        if data_types is None:
            data_types = ["rgb"]

        return CameraCfg(
            prim_path=prim_path,
            update_period=update_period,
            height=height,
            width=width,
            data_types=data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.93,
                focus_distance=400.0,
                horizontal_aperture=3.896,
                clipping_range=(0.1, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                # Left camera is offset by -47.5mm in Y direction
                pos=(0.0, -0.0475, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )

    @staticmethod
    def create_head_stereo_right_camera(
        prim_path: str = "{ENV_REGEX_NS}/Robot/H2/head_stereo_right_cam",
        height: int = 480,
        width: int = 640,
        update_period: float = 0.0,
        data_types: list[str] | None = None,
    ) -> CameraCfg:
        """Create right stereo camera configuration for the T1 head.

        This camera is positioned at the RealSense D455's right stereo camera location,
        offset by (0, 0.0475, 0) from the H2 link origin (47.5mm baseline).

        Args:
            prim_path: USD path for the camera prim.
            height: Image height in pixels. Default 480 (VGA resolution).
            width: Image width in pixels. Default 640 (VGA resolution).
            update_period: Camera update period in seconds. 0.0 = every frame.
            data_types: List of data types to capture. Default ["rgb"].

        Returns:
            CameraCfg: Camera configuration for the right stereo camera.
        """
        if data_types is None:
            data_types = ["rgb"]

        return CameraCfg(
            prim_path=prim_path,
            update_period=update_period,
            height=height,
            width=width,
            data_types=data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.93,
                focus_distance=400.0,
                horizontal_aperture=3.896,
                clipping_range=(0.1, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                # Right camera is offset by +47.5mm in Y direction
                pos=(0.0, 0.0475, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )


##
# Observation Term Configurations
##


def create_head_rgb_obs_term() -> ObsTerm:
    """Create observation term for head RGB camera.

    Returns:
        ObsTerm: Observation term configuration for RGB images.
    """
    from isaaclab_tasks.manager_based.manipulation.stack import mdp

    return ObsTerm(
        func=mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("head_rgb_cam"),
            "data_type": "rgb",
            "normalize": False,
        },
    )


def create_head_depth_obs_term() -> ObsTerm:
    """Create observation term for head depth camera.

    Returns:
        ObsTerm: Observation term configuration for depth images.
    """
    from isaaclab_tasks.manager_based.manipulation.stack import mdp

    return ObsTerm(
        func=mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("head_depth_cam"),
            "data_type": "distance_to_image_plane",
            "normalize": False,
        },
    )


##
# Example Usage
##


def get_default_t1_head_cameras(
    resolution: tuple[int, int] = (84, 84),
    include_depth: bool = True,
    include_stereo: bool = False,
) -> dict[str, CameraCfg]:
    """Get default camera configurations for T1 head.

    This is a convenience function to quickly set up common camera configurations
    for the T1 robot's head-mounted RealSense D455.

    Args:
        resolution: Camera resolution as (height, width). Default (84, 84) for
            compatibility with imitation learning policies.
        include_depth: Whether to include the depth camera. Default True.
        include_stereo: Whether to include stereo cameras. Default False.

    Returns:
        dict[str, CameraCfg]: Dictionary mapping camera names to configurations.

    Example:
        >>> cameras = get_default_t1_head_cameras(resolution=(84, 84))
        >>> scene_cfg.head_rgb_cam = cameras["head_rgb_cam"]
        >>> scene_cfg.head_depth_cam = cameras["head_depth_cam"]
    """
    cameras = {}

    # Always include RGB camera
    cameras["head_rgb_cam"] = T1HeadCameraCfg.create_head_rgb_camera(
        height=resolution[0],
        width=resolution[1],
        data_types=["rgb"],
    )

    # Optionally include depth camera
    if include_depth:
        cameras["head_depth_cam"] = T1HeadCameraCfg.create_head_depth_camera(
            height=resolution[0],
            width=resolution[1],
            data_types=["distance_to_image_plane"],
        )

    # Optionally include stereo cameras
    if include_stereo:
        cameras["head_stereo_left_cam"] = T1HeadCameraCfg.create_head_stereo_left_camera(
            height=resolution[0],
            width=resolution[1],
            data_types=["rgb"],
        )
        cameras["head_stereo_right_cam"] = T1HeadCameraCfg.create_head_stereo_right_camera(
            height=resolution[0],
            width=resolution[1],
            data_types=["rgb"],
        )

    return cameras
