"""Common utilities and configurations for T1 robot tasks."""

from .t1_camera_cfg import (
    T1HeadCameraCfg,
    create_head_depth_obs_term,
    create_head_rgb_obs_term,
    get_default_t1_head_cameras,
)
from .xr_controller_cfg import (
    create_t1_xr_controller_cfg,
    create_t1_xr_controller_full_body_cfg,
    create_t1_xr_twist_cfg,
)

__all__ = [
    # Camera configurations
    "T1HeadCameraCfg",
    "get_default_t1_head_cameras",
    "create_head_rgb_obs_term",
    "create_head_depth_obs_term",
    # XR controller configurations
    "create_t1_xr_controller_cfg",
    "create_t1_xr_controller_full_body_cfg",
    "create_t1_xr_twist_cfg",
]
