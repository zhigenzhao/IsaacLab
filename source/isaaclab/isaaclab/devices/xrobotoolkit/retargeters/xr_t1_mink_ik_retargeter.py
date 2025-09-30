# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit controller retargeter using Mink IK for T1 humanoid arm control."""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Any

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg

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


@dataclass
class XRT1MinkIKRetargeterCfg(RetargeterCfg):
    """Configuration for XRoboToolkit T1 Mink IK retargeter.

    This retargeter uses the Mink IK library to compute inverse kinematics
    for T1 humanoid arm control from XR controller poses.
    """

    xml_path: str = "source/isaaclab_assets/isaaclab_assets/robots/xmls/scene_t1_ik.xml"
    """Path to MuJoCo XML model file for T1 IK."""

    headless: bool = True
    """If True, run without MuJoCo viewer visualization."""

    ik_rate_hz: float = 100.0
    """IK solver update rate in Hz."""

    collision_avoidance_distance: float = 0.04
    """Minimum distance from collisions (meters)."""

    collision_detection_distance: float = 0.10
    """Distance at which collision avoidance activates (meters)."""

    velocity_limit_factor: float = 0.7
    """Velocity limit scaling factor for joints."""

    output_joint_positions_only: bool = False
    """If True, output only the 16 joint positions. If False, output 30 elements including hand targets."""


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

    def __init__(self, cfg: XRT1MinkIKRetargeterCfg):
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

        # Initialize MuJoCo model
        self.mj_model = mj.MjModel.from_xml_path(self._xml_path)
        self.mj_data = mj.MjData(self.mj_model)
        mj.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        # Initialize Mink IK configuration
        self.configuration = ik.Configuration(self.mj_model, self.mj_model.keyframe("home").qpos)

        # Define IK tasks for T1 robot
        self.posture_task = ik.PostureTask(self.mj_model, cost=0.05)
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

        # Start IK solver thread
        self._start_ik()

    def __del__(self):
        """Destructor to clean up IK solver thread."""
        self._stop_ik()

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
        # Initialize mocap targets
        mj.mj_forward(self.mj_model, self.mj_data)
        ik.move_mocap_to_frame(self.mj_model, self.mj_data, "left_hand_target", "left_hand", "site")
        ik.move_mocap_to_frame(self.mj_model, self.mj_data, "right_hand_target", "right_hand", "site")

        rate = RateLimiter(self._ik_rate_hz)

        while self.is_running and not self.shutdown_requested:
            with self.datalock:
                # Read mocap targets
                lh_T = ik.SE3.from_mocap_name(self.mj_model, self.mj_data, "left_hand_target")
                rh_T = ik.SE3.from_mocap_name(self.mj_model, self.mj_data, "right_hand_target")

                # Update IK tasks
                self.posture_task.set_target_from_configuration(self.configuration)
                self.lh_task.set_target(lh_T)
                self.rh_task.set_target(rh_T)

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

    def retarget(self, data: dict[str, Any]) -> torch.Tensor:
        """Convert XR controller data to T1 humanoid arm joint commands.

        Args:
            data: Dictionary containing XR controller data with enum keys

        Returns:
            torch.Tensor: Either 16-element tensor (joint positions only) or 30-element tensor
                         [upper_body_joints(16), left_hand_target(7), right_hand_target(7)]
                         depending on output_joint_positions_only configuration
        """
        from isaaclab.devices.xrobotoolkit.xr_controller import XRControllerDevice

        # Access controller data using enum keys
        left_grip = data.get(XRControllerDevice.XRControllerDeviceValues.LEFT_GRIP.value, 0.0)
        right_grip = data.get(XRControllerDevice.XRControllerDeviceValues.RIGHT_GRIP.value, 0.0)
        left_pose = data.get(XRControllerDevice.XRControllerDeviceValues.LEFT_CONTROLLER.value)
        right_pose = data.get(XRControllerDevice.XRControllerDeviceValues.RIGHT_CONTROLLER.value)

        # Convert XR controller poses to MuJoCo format [qw, qx, qy, qz, x, y, z]
        # XR format: [x, y, z, qx, qy, qz, qw]
        if left_pose is not None:
            left_pose_mj = np.array([
                left_pose[6], left_pose[3], left_pose[4], left_pose[5],  # qw, qx, qy, qz
                left_pose[0], left_pose[1], left_pose[2]  # x, y, z
            ])

        if right_pose is not None:
            right_pose_mj = np.array([
                right_pose[6], right_pose[3], right_pose[4], right_pose[5],  # qw, qx, qy, qz
                right_pose[0], right_pose[1], right_pose[2]  # x, y, z
            ])

        # Handle left hand activation
        if left_grip > 0.5 and left_pose is not None:
            if not self.lhold:
                self.reframe_mocap("left_hand_target", left_pose_mj)
                self.lhold = True
            self.sync_mocap("left_hand_target", left_pose_mj)
        else:
            self.lhold = False
            self.move_mocap_to("left_hand_target", "left_hand")

        # Handle right hand activation
        if right_grip > 0.5 and right_pose is not None:
            if not self.rhold:
                self.reframe_mocap("right_hand_target", right_pose_mj)
                self.rhold = True
            self.sync_mocap("right_hand_target", right_pose_mj)
        else:
            self.rhold = False
            self.move_mocap_to("right_hand_target", "right_hand")

        # Get IK solution
        qpos_upper = self.get_qpos_upper()

        # Return based on configuration
        if self._output_joint_positions_only:
            return torch.tensor(qpos_upper, dtype=torch.float32, device=self._sim_device)
        else:
            left_hand_target = self.get_mocap_pose_b("left_hand_target")
            right_hand_target = self.get_mocap_pose_b("right_hand_target")
            output = np.concatenate([qpos_upper, left_hand_target, right_hand_target])
            return torch.tensor(output, dtype=torch.float32, device=self._sim_device)