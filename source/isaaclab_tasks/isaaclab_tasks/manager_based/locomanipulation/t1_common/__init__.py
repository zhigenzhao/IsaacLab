"""Common utilities and configurations for T1 robot tasks."""

from .t1_camera_cfg import (
    T1HeadCameraCfg,
    create_head_depth_obs_term,
    create_head_rgb_obs_term,
    get_default_t1_head_cameras,
)

__all__ = [
    "T1HeadCameraCfg",
    "get_default_t1_head_cameras",
    "create_head_rgb_obs_term",
    "create_head_depth_obs_term",
]
