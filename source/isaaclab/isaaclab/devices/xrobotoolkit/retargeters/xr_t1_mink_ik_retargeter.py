# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit controller retargeter using Mink IK for T1 humanoid arm control."""

import numpy as np
import time
import torch
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from typing import Any

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg

# Default coordinate transformation from headset frame to world frame
R_HEADSET_TO_WORLD = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])

# Constants for state synchronization
QUAT_NORM_THRESHOLD = 1e-6  # Minimum quaternion norm for validation
SYNC_TIMEOUT_SECONDS = 0.1  # Timeout for waiting on state sync completion
SYNC_POLL_INTERVAL_SECONDS = 0.001  # Polling interval for sync completion check

# Import Mink IK dependencies
try:
    import mujoco as mj
    import mink as ik
    import threading
    from loop_rate_limiters import RateLimiter
    MINK_AVAILABLE = True
except ImportError:
    MINK_AVAILABLE = False
    print("Warning: mink, mujoco, or loop_rate_limiters not available. XRT1MinkIKRetargeter will not function.")


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(mat)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (mat[2, 1] - mat[1, 2]) / s
        y = (mat[0, 2] - mat[2, 0]) / s
        z = (mat[1, 0] - mat[0, 1]) / s
    else:
        if mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2
            w = (mat[2, 1] - mat[1, 2]) / s
            x = 0.25 * s
            y = (mat[0, 1] + mat[1, 0]) / s
            z = (mat[0, 2] + mat[2, 0]) / s
        elif mat[1, 1] > mat[2, 2]:
            s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2
            w = (mat[0, 2] - mat[2, 0]) / s
            x = (mat[0, 1] + mat[1, 0]) / s
            y = 0.25 * s
            z = (mat[1, 2] + mat[2, 1]) / s
        else:
            s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2
            w = (mat[1, 0] - mat[0, 1]) / s
            x = (mat[0, 2] + mat[2, 0]) / s
            y = (mat[1, 2] + mat[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = quat
    # Normalize quaternion
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    mat = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])
    return mat


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


class XRT1MinkIKRetargeter(RetargeterBase):
    """Retargets XR controller poses to T1 humanoid arm joint positions using Mink IK.

    This retargeter creates a MuJoCo-based IK solver running in a separate thread
    that continuously solves for T1 arm joint positions to match XR controller target poses.

    The retargeter:
    - Takes left/right controller poses from XR device
    - Runs Mink IK solver to compute T1 arm joint angles
    - Returns upper body joint positions for Isaac Lab simulation
    - Handles activation/deactivation via grip buttons

    Output format (depends on output_joint_positions_only config):
    - If output_joint_positions_only=True: 16-element tensor with upper body joint positions
    - If output_joint_positions_only=False: 30-element tensor
        - Elements 0-15: Upper body joint positions (16 joints: 2 head + 7 left arm + 7 right arm)
        - Elements 16-22: Left hand target pose in body frame [x, y, z, qw, qx, qy, qz]
        - Elements 23-29: Right hand target pose in body frame [x, y, z, qw, qx, qy, qz]
    """

    def __init__(self, cfg: "XRT1MinkIKRetargeterCfg"):
        """Initialize the T1 Mink IK retargeter.

        Args:
            cfg: Configuration for the retargeter
        """
        super().__init__(cfg)

        if not MINK_AVAILABLE:
            raise RuntimeError("Mink IK dependencies not available. Install mink, mujoco, and loop-rate-limiters.")

        self._sim_device = cfg.sim_device
        self._xml_path = cfg.xml_path
        self._headless = cfg.headless
        self._ik_rate_hz = cfg.ik_rate_hz
        self._output_joint_positions_only = cfg.output_joint_positions_only

        # Head tracking configuration
        self._enable_head_tracking = cfg.enable_head_tracking
        self._fixed_head_yaw = cfg.fixed_head_yaw
        self._fixed_head_pitch = cfg.fixed_head_pitch
        self._head_task_orientation_cost = cfg.head_task_orientation_cost
        self._head_task_position_cost = cfg.head_task_position_cost
        self._head_task_lm_damping = cfg.head_task_lm_damping

        # Initialize MuJoCo model
        self.mj_model = mj.MjModel.from_xml_path(self._xml_path)
        self.mj_data = mj.MjData(self.mj_model)
        mj.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        # Initialize Mink IK configuration
        self.configuration = ik.Configuration(self.mj_model, self.mj_model.keyframe("home").qpos)

        # Define IK tasks for T1 robot
        # Posture task maintains all joints at home position when not actively controlled
        # Head joints are now directly controlled (not via IK), so posture task helps maintain them
        posture_costs = np.array([0.05] * self.mj_model.nv)  # Uniform cost for all joints
        self.posture_task = ik.PostureTask(self.mj_model, cost=posture_costs)
        self.lh_task = ik.FrameTask(
            frame_name="left_hand",
            frame_type="site",
            position_cost=6.0,
            orientation_cost=2.0,
            lm_damping=0.03,
        )
        self.rh_task = ik.FrameTask(
            frame_name="right_hand",
            frame_type="site",
            position_cost=6.0,
            orientation_cost=2.0,
            lm_damping=0.03,
        )
        # Note: Head control is now done via direct joint angle setting, not IK task
        self.damping_task = ik.DampingTask(
            self.mj_model,
            cost=np.array([0.1] * 2 + (([0.5] * 4 + [0.1] * 3) * 2) + [0.1] * 13)
        )
        self.tasks = [self.posture_task, self.lh_task, self.rh_task, self.damping_task]

        # Define collision pairs for T1 self-collision avoidance
        self.collision_pairs = [
            (["al1", "al2", "al4", "al5", "al6", "al7", "ar1", "ar2", "ar4", "ar5", "ar6", "ar7",
              "al3_collision", "al4_collision", "ar3_collision", "ar4_collision"], ["trunk"]),
            (["al1", "al2", "al4", "al5", "al6", "al7", "ar1", "ar2","ar4", "ar5", "ar6","ar7",
              "al3_collision", "al4_collision", "ar3_collision", "ar4_collision"], ["lowerbody_box"]),
            (["al3_collision"], ["al4_collision", "left_hand_collision", "ar3_collision", "ar4_collision", "right_hand_collision"]),
            (["al4_collision"], ["al3_collision", "left_hand_collision", "ar3_collision", "ar4_collision", "right_hand_collision"]),
            (["left_hand_collision"], ["al3_collision", "al4_collision", "ar3_collision", "ar4_collision", "right_hand_collision"]),
            (["ar3_collision"], ["al4_collision", "left_hand_collision", "al3_collision", "ar4_collision", "right_hand_collision"]),
            (["ar4_collision"], ["al4_collision", "left_hand_collision", "ar3_collision", "al3_collision", "right_hand_collision"]),
            (["right_hand_collision"], ["al4_collision", "left_hand_collision", "ar3_collision", "ar4_collision", "al3_collision"]),
        ]

        # Define IK limits for T1 joints
        factor = cfg.velocity_limit_factor
        self.limits = [
            ik.ConfigurationLimit(self.mj_model, min_distance_from_limits=0.1),
            ik.CollisionAvoidanceLimit(
                self.mj_model,
                self.collision_pairs,
                minimum_distance_from_collisions=cfg.collision_avoidance_distance,
                collision_detection_distance=cfg.collision_detection_distance,
            ),
            ik.VelocityLimit(
                self.mj_model,
                {
                    "Left_Shoulder_Pitch": 5.0 * factor,
                    "Left_Shoulder_Roll": 5.0 * factor,
                    "Left_Elbow_Pitch": 5.0 * factor,
                    "Left_Elbow_Yaw": 5.0 * factor,
                    "Left_Wrist_Pitch": 2.5,
                    "Left_Wrist_Yaw": 2.5,
                    "Left_Hand_Roll": 2.5,
                    "Right_Shoulder_Pitch": 5.0 * factor,
                    "Right_Shoulder_Roll": 5.0 * factor,
                    "Right_Elbow_Pitch": 5.0 * factor,
                    "Right_Elbow_Yaw": 5.0 * factor,
                    "Right_Wrist_Pitch": 2.5,
                    "Right_Wrist_Yaw": 2.5,
                    "Right_Hand_Roll": 2.5,
                    "AAHead_yaw": 6.0,
                    "Head_pitch": 5.0
                }
            ),
        ]

        # Thread synchronization
        self.datalock = threading.RLock()
        self.is_running = False
        self.is_ready = False
        self.is_solving = True
        self.shutdown_requested = False

        # Viewer setup
        if not self._headless:
            import mujoco.viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.mj_model, self.mj_data)
        else:
            self.viewer = None

        # Mocap tracking state
        self.synced_mocap = {}
        self.lhold = False
        self.rhold = False
        self.htrack = False  # Head tracking activation state

        # Head joint angles (directly controlled, not via IK)
        # Initialize with fixed values (will be updated if head tracking is enabled)
        self.desired_head_yaw = self._fixed_head_yaw
        self.desired_head_pitch = self._fixed_head_pitch

        # Measured joint positions from simulation (for state sync)
        self.measured_joint_positions = None
        self.force_sync = False
        self.sync_complete = True  # Flag to indicate sync is complete

        # Reference frame for relative control
        self._reference_frame = cfg.reference_frame

        # Motion tracker configuration and state
        self._motion_tracker_config = cfg.motion_tracker_config
        self._motion_tracker_task_weight = cfg.motion_tracker_task_weight
        self._arm_length_scale_factor = cfg.arm_length_scale_factor
        self.tracker_tasks = {}  # Dict of Mink position tasks for motion trackers

        # Setup motion tracker tasks if configured
        if self._motion_tracker_config:
            print(f"[XRT1MinkIKRetargeter] Motion tracker config: {self._motion_tracker_config}")
            self._setup_motion_tracker_tasks()
        else:
            print(f"[XRT1MinkIKRetargeter] No motion tracker configured")

        # Start IK solver thread
        self._start_ik()

    def __del__(self):
        """Destructor to clean up IK solver thread."""
        self._stop_ik()

    def _setup_motion_tracker_tasks(self):
        """Setup Mink IK position tasks for motion trackers.

        Creates position-only frame tasks that track mocap bodies representing
        elbow target positions. The mocap bodies are updated based on the relative
        position between the motion tracker and controller.
        """
        with self.datalock:
            for arm_name, tracker_config in self._motion_tracker_config.items():
                link_target = tracker_config["link_target"]
                serial = tracker_config["serial"]

                # Determine mocap body name based on arm
                if arm_name == "left_arm":
                    mocap_name = "left_elbow_target"
                elif arm_name == "right_arm":
                    mocap_name = "right_elbow_target"
                else:
                    print(f"Warning: Unknown arm name '{arm_name}'. Skipping motion tracker setup.")
                    continue

                # Verify the link exists in the model
                try:
                    self.mj_model.site(link_target).id
                except KeyError:
                    print(f"Warning: Motion tracker link site '{link_target}' not found in model. Skipping.")
                    continue

                # Initialize mocap body to current elbow position
                ik.move_mocap_to_frame(self.mj_model, self.mj_data, mocap_name, link_target, "site")

                # Create position-only frame task that follows the mocap body
                tracker_task = ik.FrameTask(
                    frame_name=link_target,
                    frame_type="site",
                    position_cost=self._motion_tracker_task_weight,
                    orientation_cost=0.0,  # Position only
                    lm_damping=0.03,
                )

                # Set task target from mocap body
                mocap_se3 = ik.SE3.from_mocap_name(self.mj_model, self.mj_data, mocap_name)
                tracker_task.set_target(mocap_se3)

                # Add task to solver
                self.tracker_tasks[arm_name] = tracker_task
                self.tasks.append(tracker_task)

                print(f"Motion tracker task created: {arm_name} -> {link_target} (serial: {serial}, mocap: {mocap_name})")

    def _start_ik(self):
        """Start the IK solver thread."""
        if self.is_running:
            return
        self.is_running = True
        self.is_ready = False
        self.ik_thread = threading.Thread(target=self._solve_ik_loop, name="T1MinkIKThread")
        self.ik_thread.daemon = True
        self.ik_thread.start()

    def _stop_ik(self):
        """Stop the IK solver thread."""
        print("Stopping T1 Mink IK solver...")
        self.shutdown_requested = True
        self.is_running = False

        if hasattr(self, 'ik_thread') and self.ik_thread.is_alive():
            self.ik_thread.join(timeout=2.0)
            if self.ik_thread.is_alive():
                print("Warning: T1 Mink IK thread did not stop cleanly")

        if self.viewer is not None:
            try:
                self.viewer.close()
            except:
                pass

    def _solve_ik_loop(self):
        """Main IK solver loop running in separate thread."""
        # Initialize mocap targets (hands only, head is now controlled directly)
        mj.mj_forward(self.mj_model, self.mj_data)
        ik.move_mocap_to_frame(self.mj_model, self.mj_data, "left_hand_target", "left_hand", "site")
        ik.move_mocap_to_frame(self.mj_model, self.mj_data, "right_hand_target", "right_hand", "site")

        rate = RateLimiter(self._ik_rate_hz)

        while self.is_running and not self.shutdown_requested:
            with self.datalock:
                # Force sync if requested (e.g., on reset or grip press)
                if self.force_sync and self.measured_joint_positions is not None:
                    self.set_qpos_upper(self.measured_joint_positions)
                    # Update mocap targets to match the actual positions/orientations in the synced state
                    # This prevents jumps when reframe_mocap calculates the offset
                    ik.move_mocap_to_frame(self.mj_model, self.mj_data, "left_hand_target", "left_hand", "site")
                    ik.move_mocap_to_frame(self.mj_model, self.mj_data, "right_hand_target", "right_hand", "site")
                    self.force_sync = False
                    self.sync_complete = True

                # Read mocap targets (hands only, head is now controlled directly)
                lh_T = ik.SE3.from_mocap_name(self.mj_model, self.mj_data, "left_hand_target")
                rh_T = ik.SE3.from_mocap_name(self.mj_model, self.mj_data, "right_hand_target")

                # Update IK tasks (no head task anymore)
                self.posture_task.set_target_from_configuration(self.configuration)
                self.lh_task.set_target(lh_T)
                self.rh_task.set_target(rh_T)

                # Update motion tracker tasks from their mocap bodies
                if self._motion_tracker_config:
                    for arm_name in self.tracker_tasks:
                        if arm_name == "left_arm":
                            elbow_mocap_name = "left_elbow_target"
                        elif arm_name == "right_arm":
                            elbow_mocap_name = "right_elbow_target"
                        else:
                            continue

                        elbow_T = ik.SE3.from_mocap_name(self.mj_model, self.mj_data, elbow_mocap_name)
                        self.tracker_tasks[arm_name].set_target(elbow_T)

                # Solve IK
                if self.is_solving:
                    vel = ik.solve_ik(
                        self.configuration,
                        self.tasks,
                        rate.dt,
                        "daqp",
                        1e-2,
                        safety_break=True,
                        limits=self.limits,
                    )
                    self.configuration.integrate_inplace(vel, rate.dt)
                    self.mj_data.qpos[:] = self.configuration.q
                else:
                    self.configuration = ik.Configuration(self.mj_model, self.mj_data.qpos[:])

                # Apply head joint angles directly (overriding IK solution for head)
                yaw_qpos_idx = self.mj_model.joint('AAHead_yaw').qposadr[0]
                pitch_qpos_idx = self.mj_model.joint('Head_pitch').qposadr[0]
                self.mj_data.qpos[yaw_qpos_idx] = self.desired_head_yaw
                self.mj_data.qpos[pitch_qpos_idx] = self.desired_head_pitch
                self.configuration.q[yaw_qpos_idx] = self.desired_head_yaw
                self.configuration.q[pitch_qpos_idx] = self.desired_head_pitch

                mj.mj_forward(self.mj_model, self.mj_data)

            if self.viewer is not None:
                with self.viewer.lock():
                    self.viewer.sync()

            self.is_ready = True
            rate.sleep()

        self.is_running = False
        self.is_ready = False

    def reframe_mocap(self, name: str, wxyz_xyz: np.ndarray, relative_site_name: str = "world"):
        """Establish reference frame for mocap target tracking.

        Args:
            name: Name of mocap body
            wxyz_xyz: Target pose [qw, qx, qy, qz, x, y, z]
            relative_site_name: Site to use as reference frame
        """
        if not self.is_ready:
            return

        pos = wxyz_xyz[4:]
        quat = wxyz_xyz[:4]

        site_id = self.mj_model.site(relative_site_name).id
        site_xpos = self.mj_data.site_xpos[site_id]
        site_xmat = self.mj_data.site_xmat[site_id].reshape(3, 3)

        mocap_id = self.mj_model.body(name).mocapid[0]

        # Get current mocap pose in world frame
        mocap_xpos_W = self.mj_data.mocap_pos[mocap_id].copy()
        mocap_quat_W = self.mj_data.mocap_quat[mocap_id].copy()

        # Transform to site frame
        site_quat = mat_to_quat(site_xmat)
        site_quat_inv = np.array([site_quat[0], -site_quat[1], -site_quat[2], -site_quat[3]])

        # Compute offsets
        pos_rel_to_site = site_xmat.T @ (mocap_xpos_W - site_xpos)
        pos_offset = pos_rel_to_site - pos

        current_quat_rel_to_site = quat_multiply(site_quat_inv, mocap_quat_W)
        desired_quat_inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
        quat_offset = quat_multiply(desired_quat_inv, current_quat_rel_to_site)

        self.synced_mocap[name] = {
            "pos_offset": pos_offset,
            "quat_offset": quat_offset
        }

    def sync_mocap(self, name: str, wxyz_xyz: np.ndarray, relative_site_name: str = "world"):
        """Update mocap target with offset tracking.

        Args:
            name: Name of mocap body
            wxyz_xyz: Target pose [qw, qx, qy, qz, x, y, z]
            relative_site_name: Site to use as reference frame
        """
        if not self.is_ready:
            return

        if name not in self.synced_mocap:
            self.reframe_mocap(name, wxyz_xyz, relative_site_name)
            return

        mocap_id = self.mj_model.body(name).mocapid[0]
        site_id = self.mj_model.site(relative_site_name).id
        site_xpos = self.mj_data.site_xpos[site_id]
        site_xmat = self.mj_data.site_xmat[site_id].reshape(3, 3)

        pos_offset = self.synced_mocap[name]["pos_offset"]
        quat_offset = self.synced_mocap[name]["quat_offset"]

        pos = wxyz_xyz[4:]
        quat = wxyz_xyz[:4]

        # Apply offsets
        pos_corrected = pos + pos_offset
        relative_quat_corrected = quat_multiply(quat, quat_offset)

        # Convert to world frame
        mocap_xpos_W = site_xpos + site_xmat @ pos_corrected
        site_quat = mat_to_quat(site_xmat)
        mocap_quat_W = quat_multiply(site_quat, relative_quat_corrected)

        self.mj_data.mocap_pos[mocap_id] = mocap_xpos_W
        self.mj_data.mocap_quat[mocap_id] = mocap_quat_W

    def move_mocap_to(self, name: str, target_site_name: str):
        """Move mocap target to match a site's current pose.

        Args:
            name: Name of mocap body
            target_site_name: Name of site to match
        """
        ik.move_mocap_to_frame(self.mj_model, self.mj_data, name, target_site_name, "site")

    def _transform_world_to_site(self, wxyz_xyz_world: np.ndarray, site_name: str) -> np.ndarray:
        """Transform a pose from world frame to site-relative frame.

        Args:
            wxyz_xyz_world: Pose in world frame [qw, qx, qy, qz, x, y, z]
            site_name: Name of the site to use as reference frame

        Returns:
            Pose in site-relative frame [qw, qx, qy, qz, x, y, z]

        Raises:
            ValueError: If site_name does not exist in the model
        """
        # Validate site exists
        try:
            site_id = self.mj_model.site(site_name).id
        except KeyError:
            raise ValueError(f"Site '{site_name}' not found in MuJoCo model")

        site_xpos = self.mj_data.site_xpos[site_id]
        site_xmat = self.mj_data.site_xmat[site_id].reshape(3, 3)

        # Extract world pose
        quat_world = wxyz_xyz_world[:4]
        pos_world = wxyz_xyz_world[4:]

        # Transform position to site frame
        pos_rel = site_xmat.T @ (pos_world - site_xpos)

        # Transform orientation to site frame
        site_quat = mat_to_quat(site_xmat)
        site_quat_inv = np.array([site_quat[0], -site_quat[1], -site_quat[2], -site_quat[3]])
        quat_rel = quat_multiply(site_quat_inv, quat_world)

        return np.concatenate([quat_rel, pos_rel])

    def _transform_xr_pose_to_reference_frame(self, xr_pose: np.ndarray) -> np.ndarray:
        """Transform XR controller pose from headset frame to reference frame.

        This method handles the full transformation chain:
        1. Headset frame → World frame (using R_HEADSET_TO_WORLD)
        2. World frame → Reference frame (using _transform_world_to_site)

        Args:
            xr_pose: Pose in XR format [x, y, z, qx, qy, qz, qw] in headset frame

        Returns:
            Pose in MuJoCo format [qw, qx, qy, qz, x, y, z] in reference frame
        """
        # Extract position and quaternion from XR format
        pos_headset = xr_pose[:3]
        quat_xr = xr_pose[3:]  # [qx, qy, qz, qw]
        quat_headset = np.array([quat_xr[3], quat_xr[0], quat_xr[1], quat_xr[2]])  # [qw, qx, qy, qz]

        # Check for valid quaternion (non-zero norm)
        quat_norm = np.linalg.norm(quat_headset)
        if quat_norm < QUAT_NORM_THRESHOLD:
            # Invalid quaternion, use identity
            quat_headset = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            quat_headset = quat_headset / quat_norm  # Normalize

        # Transform position from headset to world
        pos_world = R_HEADSET_TO_WORLD @ pos_headset

        # Transform orientation from headset to world: R_quat * headset_quat * R_quat_conj
        R_quat_scipy = Rotation.from_matrix(R_HEADSET_TO_WORLD).as_quat()  # [x, y, z, w]
        R_quat = np.array([R_quat_scipy[3], R_quat_scipy[0], R_quat_scipy[1], R_quat_scipy[2]])  # [w, x, y, z]
        R_quat_conj = np.array([R_quat[0], -R_quat[1], -R_quat[2], -R_quat[3]])
        quat_world = quat_multiply(quat_multiply(R_quat, quat_headset), R_quat_conj)

        # Create world frame pose in MuJoCo format
        pose_mj_world = np.concatenate([quat_world, pos_world])

        # Transform to reference frame
        return self._transform_world_to_site(pose_mj_world, self._reference_frame)

    def _extract_head_joint_angles_from_headset(self, headset_pose: np.ndarray) -> tuple[float, float]:
        """Extract yaw and pitch joint angles from headset orientation in world frame.

        This method transforms the headset orientation to world frame and extracts
        the yaw and pitch angles for direct joint control (no IK).

        Args:
            headset_pose: Headset pose [x, y, z, qx, qy, qz, qw] from XR device

        Returns:
            Tuple of (yaw, pitch) in radians for AAHead_yaw and Head_pitch joints

        Raises:
            ValueError: If quaternion has zero norm (invalid headset data)
        """
        # Extract quaternion from XR format [x, y, z, qx, qy, qz, qw]
        quat_xr = headset_pose[3:]  # [qx, qy, qz, qw]
        quat_headset = np.array([quat_xr[3], quat_xr[0], quat_xr[1], quat_xr[2]])  # [qw, qx, qy, qz]

        # Validate quaternion
        quat_norm = np.linalg.norm(quat_headset)
        if quat_norm < QUAT_NORM_THRESHOLD:
            raise ValueError(f"Invalid headset quaternion (norm={quat_norm:.6f}). Headset data not ready.")

        quat_headset = quat_headset / quat_norm  # Normalize

        # Transform orientation from headset to world frame: R_quat * headset_quat * R_quat_conj
        R_quat_scipy = Rotation.from_matrix(R_HEADSET_TO_WORLD).as_quat()  # [x, y, z, w]
        R_quat = np.array([R_quat_scipy[3], R_quat_scipy[0], R_quat_scipy[1], R_quat_scipy[2]])  # [w, x, y, z]
        R_quat_conj = np.array([R_quat[0], -R_quat[1], -R_quat[2], -R_quat[3]])
        quat_world = quat_multiply(quat_multiply(R_quat, quat_headset), R_quat_conj)

        # Extract Euler angles (XYZ intrinsic convention)
        euler_angles = Rotation.from_quat([quat_world[1], quat_world[2], quat_world[3], quat_world[0]]).as_euler('xyz', degrees=False)
        yaw = euler_angles[2]   # Rotation around Z-axis -> AAHead_yaw
        pitch = euler_angles[1]  # Rotation around Y-axis -> Head_pitch

        # Apply joint limits
        # AAHead_yaw: range="-1.57 1.57" (-90° to +90°)
        # Head_pitch: range="-0.35 1.22" (-20° to +70°)
        yaw = np.clip(yaw, -1.57, 1.57)
        pitch = np.clip(pitch, -0.35, 1.22)

        return yaw, pitch

    def set_head_joint_angles(self, yaw: float, pitch: float):
        """Set desired head joint angles (applied in IK loop).

        This method stores the desired head joint angles, which are then applied
        in the IK solver loop after each IK solve.

        Args:
            yaw: Target yaw angle in radians for AAHead_yaw joint
            pitch: Target pitch angle in radians for Head_pitch joint
        """
        with self.datalock:
            self.desired_head_yaw = yaw
            self.desired_head_pitch = pitch

    def get_qpos_upper(self) -> np.ndarray:
        """Get current T1 upper body joint positions.

        Returns:
            Array of 16 upper body joint positions (2 head + 7 left arm + 7 right arm)
        """
        with self.datalock:
            res = np.zeros(16)
            for i, jnt_name in enumerate([
                "AAHead_yaw", "Head_pitch",
                "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
                "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
                "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
                "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
            ]):
                res[i] = self.mj_data.joint(jnt_name).qpos
            return res

    def set_qpos_upper(self, joint_positions: np.ndarray):
        """Update Mink internal state from measured joint positions.

        This syncs the internal MuJoCo simulation state with the actual Isaac Lab
        simulation state, preventing drift between the IK solver and the real robot.

        Args:
            joint_positions: Array of 16 upper body joint positions (2 head + 7 left arm + 7 right arm)
        """
        if joint_positions is None or len(joint_positions) != 16:
            return

        with self.datalock:
            joint_names = [
                "AAHead_yaw", "Head_pitch",
                "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
                "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
                "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
                "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
            ]

            for i, jnt_name in enumerate(joint_names):
                self.mj_data.joint(jnt_name).qpos = joint_positions[i]

            # Update configuration to match new state
            self.configuration = ik.Configuration(self.mj_model, self.mj_data.qpos[:])

            # Forward kinematics to update derived quantities
            mj.mj_forward(self.mj_model, self.mj_data)

    def reset(self, joint_positions: np.ndarray | None = None):
        """Reset IK solver state to match current simulation state.

        This is typically called on environment reset to ensure the Mink IK solver
        starts from the correct initial state.

        Args:
            joint_positions: Optional array of 16 upper body joint positions. If None, resets to home position.
        """
        with self.datalock:
            # Update joint positions if provided
            if joint_positions is not None:
                joint_names = [
                    "AAHead_yaw", "Head_pitch",
                    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
                    "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
                    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
                    "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
                ]
                for i, jnt_name in enumerate(joint_names):
                    self.mj_data.joint(jnt_name).qpos = joint_positions[i]
                self.configuration = ik.Configuration(self.mj_model, self.mj_data.qpos[:])
            else:
                # Reset to home keyframe
                mj.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
                self.configuration = ik.Configuration(self.mj_model, self.mj_model.keyframe("home").qpos)

            # Reset mocap targets to current positions/orientations (hands only, head is directly controlled)
            mj.mj_forward(self.mj_model, self.mj_data)
            ik.move_mocap_to_frame(self.mj_model, self.mj_data, "left_hand_target", "left_hand", "site")
            ik.move_mocap_to_frame(self.mj_model, self.mj_data, "right_hand_target", "right_hand", "site")

            # Reset elbow mocap bodies if motion trackers are configured
            if self._motion_tracker_config:
                if "left_arm" in self._motion_tracker_config:
                    link_target = self._motion_tracker_config["left_arm"]["link_target"]
                    ik.move_mocap_to_frame(self.mj_model, self.mj_data, "left_elbow_target", link_target, "site")
                if "right_arm" in self._motion_tracker_config:
                    link_target = self._motion_tracker_config["right_arm"]["link_target"]
                    ik.move_mocap_to_frame(self.mj_model, self.mj_data, "right_elbow_target", link_target, "site")

            # Clear tracking state
            self.synced_mocap = {}
            self.lhold = False
            self.rhold = False
            self.htrack = False

            # Clear force sync flag
            self.force_sync = False

    def get_mocap_pose_b(self, name: str) -> np.ndarray:
        """Get mocap pose in body frame.

        Args:
            name: Name of mocap body

        Returns:
            Pose array [x, y, z, qw, qx, qy, qz] in body frame
        """
        mocap_id = self.mj_model.body(name).mocapid[0]
        site = self.mj_data.site("imu")
        g_wb = ik.SE3(np.concatenate([mat_to_quat(site.xmat.reshape(3, 3)), site.xpos]))
        g_wm = ik.SE3(np.concatenate([self.mj_data.mocap_quat[mocap_id], self.mj_data.mocap_pos[mocap_id]]))
        g_bm = g_wb.inverse().multiply(g_wm)
        return np.concatenate([g_bm.wxyz_xyz[4:], g_bm.wxyz_xyz[:4]])

    def _update_motion_tracker_tasks(self, motion_tracker_data: dict, left_active: bool, right_active: bool, data: dict[str, Any] = None):
        """Update elbow mocap bodies based on motion tracker data.

        Continuously calculates the offset between tracker and controller positions,
        then applies this offset (scaled) to the hand target mocap position to
        determine the elbow target position.

        Args:
            motion_tracker_data: Dictionary of motion tracker data {serial: {"pose": [x,y,z,qx,qy,qz,qw]}}
            left_active: Whether left arm control is active (grip pressed)
            right_active: Whether right arm control is active (grip pressed)
            data: Full device data dictionary containing controller poses
        """
        from isaaclab.devices.xrobotoolkit.xr_controller import XRControllerDevice

        if not self._motion_tracker_config or data is None:
            return

        # Map arm names to activation status and hand mocap names
        arm_info = {
            "left_arm": {
                "active": left_active,
                "hand_mocap": "left_hand_target",
                "elbow_mocap": "left_elbow_target",
                "elbow_site": None  # Will be filled from config
            },
            "right_arm": {
                "active": right_active,
                "hand_mocap": "right_hand_target",
                "elbow_mocap": "right_elbow_target",
                "elbow_site": None  # Will be filled from config
            }
        }

        with self.datalock:
            for arm_name, tracker_config in self._motion_tracker_config.items():
                serial = tracker_config["serial"]
                link_target = tracker_config["link_target"]

                if arm_name not in arm_info:
                    continue

                arm_data = arm_info[arm_name]
                arm_data["elbow_site"] = link_target

                # If arm is inactive, move elbow mocap to current actual elbow position
                if not arm_data["active"]:
                    ik.move_mocap_to_frame(
                        self.mj_model, self.mj_data,
                        arm_data["elbow_mocap"], link_target, "site"
                    )
                    continue

                # Skip if tracker not in data
                if serial not in motion_tracker_data:
                    # If no tracker data, keep elbow at current position
                    ik.move_mocap_to_frame(
                        self.mj_model, self.mj_data,
                        arm_data["elbow_mocap"], link_target, "site"
                    )
                    continue

                # Get tracker pose transformed to reference frame
                tracker_pose_xr = motion_tracker_data[serial]["pose"]
                tracker_pose_mj = self._transform_xr_pose_to_reference_frame(tracker_pose_xr)
                tracker_xyz = tracker_pose_mj[4:]  # [qw,qx,qy,qz,x,y,z] -> [x,y,z]

                # Get controller pose transformed to reference frame (not mocap target!)
                if arm_name == "left_arm":
                    controller_pose_xr = data.get(XRControllerDevice.XRControllerDeviceValues.LEFT_CONTROLLER.value)
                elif arm_name == "right_arm":
                    controller_pose_xr = data.get(XRControllerDevice.XRControllerDeviceValues.RIGHT_CONTROLLER.value)
                else:
                    continue

                if controller_pose_xr is None:
                    continue

                controller_pose_mj = self._transform_xr_pose_to_reference_frame(controller_pose_xr)
                controller_xyz = controller_pose_mj[4:]  # [qw,qx,qy,qz,x,y,z] -> [x,y,z]

                # Calculate offset in reference frame
                offset = tracker_xyz - controller_xyz

                # Get hand mocap target position and add scaled offset
                hand_mocap_id = self.mj_model.body(arm_data["hand_mocap"]).mocapid[0]
                hand_target_xyz = self.mj_data.mocap_pos[hand_mocap_id].copy()

                # Calculate elbow target: hand target + scaled offset
                elbow_target_xyz = hand_target_xyz + offset * self._arm_length_scale_factor

                # Update elbow mocap body
                elbow_mocap_id = self.mj_model.body(arm_data["elbow_mocap"]).mocapid[0]
                self.mj_data.mocap_pos[elbow_mocap_id] = elbow_target_xyz

    def retarget(self, data: dict[str, Any]) -> torch.Tensor:
        """Convert XR controller data to T1 humanoid arm joint commands.

        Args:
            data: Dictionary containing XR controller data with enum keys.
                  Can optionally include "measured_joint_positions" key with 16-element array
                  of current upper body joint positions from Isaac Lab simulation.

        Returns:
            torch.Tensor: Either 16-element tensor (joint positions only) or 30-element tensor
                         [upper_body_joints(16), left_hand_target(7), right_hand_target(7)]
                         depending on output_joint_positions_only configuration
        """
        from isaaclab.devices.xrobotoolkit.xr_controller import XRControllerDevice

        # Store measured joint positions from simulation if provided
        # This enables state sync to prevent drift between Mink IK and Isaac Lab
        measured_positions = data.get("measured_joint_positions")
        if measured_positions is not None:
            # Convert to numpy if it's a tensor
            if isinstance(measured_positions, torch.Tensor):
                measured_positions = measured_positions.cpu().numpy()
            with self.datalock:
                self.measured_joint_positions = measured_positions.copy()

        # Access controller data using enum keys
        left_grip = data.get(XRControllerDevice.XRControllerDeviceValues.LEFT_GRIP.value, 0.0)
        right_grip = data.get(XRControllerDevice.XRControllerDeviceValues.RIGHT_GRIP.value, 0.0)
        left_pose = data.get(XRControllerDevice.XRControllerDeviceValues.LEFT_CONTROLLER.value)
        right_pose = data.get(XRControllerDevice.XRControllerDeviceValues.RIGHT_CONTROLLER.value)

        # Transform XR controller poses to reference frame
        # XR format: [x, y, z, qx, qy, qz, qw] in headset frame
        # Output: [qw, qx, qy, qz, x, y, z] in reference frame (e.g., trunk)
        if left_pose is not None:
            left_pose_mj = self._transform_xr_pose_to_reference_frame(left_pose)

        if right_pose is not None:
            right_pose_mj = self._transform_xr_pose_to_reference_frame(right_pose)

        # Handle left hand activation
        if left_grip > 0.5 and left_pose is not None:
            if not self.lhold:
                # Request state sync on grip press transition
                # The sync will be performed by the IK solver thread
                if self.measured_joint_positions is not None:
                    self.sync_complete = False
                    self.force_sync = True
                    # Wait for sync to complete before reframing mocap
                    # This ensures the offset is calculated with the synced state
                    start_time = time.time()
                    while not self.sync_complete and (time.time() - start_time) < SYNC_TIMEOUT_SECONDS:
                        time.sleep(SYNC_POLL_INTERVAL_SECONDS)
                self.reframe_mocap("left_hand_target", left_pose_mj, relative_site_name=self._reference_frame)
                self.lhold = True
            self.sync_mocap("left_hand_target", left_pose_mj, relative_site_name=self._reference_frame)
        else:
            self.lhold = False
            self.move_mocap_to("left_hand_target", "left_hand")

        # Handle right hand activation
        if right_grip > 0.5 and right_pose is not None:
            if not self.rhold:
                # Request state sync on grip press transition
                # The sync will be performed by the IK solver thread
                if self.measured_joint_positions is not None:
                    self.sync_complete = False
                    self.force_sync = True
                    # Wait for sync to complete before reframing mocap
                    # This ensures the offset is calculated with the synced state
                    start_time = time.time()
                    while not self.sync_complete and (time.time() - start_time) < SYNC_TIMEOUT_SECONDS:
                        time.sleep(SYNC_POLL_INTERVAL_SECONDS)
                self.reframe_mocap("right_hand_target", right_pose_mj, relative_site_name=self._reference_frame)
                self.rhold = True
            self.sync_mocap("right_hand_target", right_pose_mj, relative_site_name=self._reference_frame)
        else:
            self.rhold = False
            self.move_mocap_to("right_hand_target", "right_hand")

        # Update motion tracker targets (elbow tracking)
        if self._motion_tracker_config:
            motion_tracker_data = data.get(XRControllerDevice.XRControllerDeviceValues.MOTION_TRACKERS.value, {})
            left_active = left_grip > 0.5 and left_pose is not None
            right_active = right_grip > 0.5 and right_pose is not None

            self._update_motion_tracker_tasks(motion_tracker_data, left_active, right_active, data)

        # Handle head orientation tracking via direct joint angle control
        if self._enable_head_tracking:
            headset_pose = data.get(XRControllerDevice.XRControllerDeviceValues.HEADSET.value)
            if headset_pose is not None:
                try:
                    # Extract yaw and pitch angles from headset orientation
                    yaw, pitch = self._extract_head_joint_angles_from_headset(headset_pose)

                    # Directly set head joint angles (bypassing IK)
                    self.set_head_joint_angles(yaw, pitch)

                    # Mark head tracking as active
                    if not self.htrack:
                        print("Head tracking activated (direct joint control)")
                        self.htrack = True

                except ValueError as e:
                    # Invalid headset data - keep head at fixed angles
                    if self.htrack:
                        print(f"Head tracking deactivated: {e}")
                        self.htrack = False
                    # Reset to fixed angles
                    self.set_head_joint_angles(self._fixed_head_yaw, self._fixed_head_pitch)
        else:
            # Head tracking disabled - keep head at fixed angles
            if self.htrack:
                print("Head tracking disabled - using fixed angles")
                self.htrack = False
            # Ensure head stays at fixed angles
            self.set_head_joint_angles(self._fixed_head_yaw, self._fixed_head_pitch)

        # Get IK solution (head joints are directly set, not from IK)
        qpos_upper = self.get_qpos_upper()

        # Return based on configuration
        if self._output_joint_positions_only:
            return torch.tensor(qpos_upper, dtype=torch.float32, device=self._sim_device)
        else:
            left_hand_target = self.get_mocap_pose_b("left_hand_target")
            right_hand_target = self.get_mocap_pose_b("right_hand_target")
            output = np.concatenate([qpos_upper, left_hand_target, right_hand_target])
            return torch.tensor(output, dtype=torch.float32, device=self._sim_device)


@dataclass
class XRT1MinkIKRetargeterCfg(RetargeterCfg):
    """Configuration for T1 Mink IK retargeter.

    This retargeter uses Mink IK to compute T1 arm joint positions from XR controller poses.
    """

    xml_path: str = ""
    """Path to T1 MuJoCo XML model file."""

    headless: bool = True
    """If True, run without MuJoCo viewer visualization."""

    ik_rate_hz: float = 200.0
    """IK solver update rate in Hz."""

    output_joint_positions_only: bool = True
    """If True, output only joint positions. If False, include hand target poses."""

    enable_head_tracking: bool = False
    """If True, track headset orientation for head joint control."""

    fixed_head_yaw: float = 0.0
    """Fixed yaw angle for head when not tracking (radians)."""

    fixed_head_pitch: float = 1.0472
    """Fixed pitch angle for head when not tracking (radians, ~60 degrees down)."""

    head_task_orientation_cost: float = 2.0
    """Orientation cost for head IK task."""

    head_task_position_cost: float = 0.0
    """Position cost for head IK task (typically 0 for orientation-only tracking)."""

    head_task_lm_damping: float = 0.03
    """Levenberg-Marquardt damping for head IK task."""

    velocity_limit_factor: float = 1.0
    """Scaling factor for joint velocity limits."""

    collision_avoidance_distance: float = 0.02
    """Minimum distance from collisions in meters."""

    collision_detection_distance: float = 0.05
    """Distance at which collision avoidance activates in meters."""

    reference_frame: str = "trunk"
    """Site name to use as reference frame for relative control."""

    motion_tracker_config: dict | None = None
    """Configuration for motion trackers (elbow tracking). Format:
    {"left_arm": {"link_target": "site_name", "serial": "tracker_serial"}, ...}"""

    motion_tracker_task_weight: float = 1.0
    """Position task weight for motion tracker targets."""

    arm_length_scale_factor: float = 1.0
    """Scale factor for arm length (tracker offset scaling)."""

    retargeter_type: type[RetargeterBase] = XRT1MinkIKRetargeter