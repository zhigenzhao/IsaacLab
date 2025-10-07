# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common physics configuration for T1 manipulation tasks."""

from isaaclab.sim import RigidBodyPropertiesCfg

# Standard cube/object properties for stacking/manipulation
MANIPULATION_OBJECT_PROPERTIES = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

# PhysX solver settings for manipulation
MANIPULATION_PHYSX_SETTINGS = {
    "solver_type": 1,
    "bounce_threshold_velocity": 0.2,
    "gpu_found_lost_aggregate_pairs_capacity": 1024 * 1024 * 4,
    "gpu_total_aggregate_pairs_capacity": 16 * 1024,
    "friction_correlation_distance": 0.00625,
}
