# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit controller retargeter using TWIST policy for full-body humanoid control."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import threading
import torch
import yaml

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from .xr_gmr_retargeter import GMROutputFormat, XRGMRRetargeter, XRGMRRetargeterCfg

# Import rate limiter for threading
try:
    from loop_rate_limiters import RateLimiter
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False

# Import ONNX runtime for policy inference
try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX runtime not available. XRTwistRetargeter will not function.")


class TwistOutputFormat(Enum):
    """Output format options for TWIST retargeter.

    Determines what data is returned from the retargeter:
    - ABSOLUTE: Joint position targets (policy offset + default angles)
    - OFFSET: Raw policy output (position offsets from default)
    """

    ABSOLUTE = "absolute"
    """Joint position targets: offset + default_joint_positions"""

    OFFSET = "offset"
    """Raw position offsets from default pose"""


def euler_from_quat(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat: Quaternion in [w, x, y, z] format

    Returns:
        Euler angles [roll, pitch, yaw] in radians
    """
    qw, qx, qy, qz = quat
    euler = np.zeros(3)

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    euler[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        euler[1] = np.copysign(np.pi / 2, sinp)
    else:
        euler[1] = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    euler[2] = np.arctan2(siny_cosp, cosy_cosp)

    return euler


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse of quaternion.

    Args:
        q: Quaternion(s) in [x, y, z, w] format, shape (..., 4)
        v: Vector(s) to rotate, shape (..., 3)

    Returns:
        Rotated vector(s), shape (..., 3)
    """
    q = np.asarray(q)
    v = np.asarray(v)

    q_w = q[..., -1]  # w
    q_vec = q[..., :3]  # x, y, z

    a = v * (2.0 * q_w**2 - 1.0)[..., np.newaxis]
    b = np.cross(q_vec, v) * (2.0 * q_w)[..., np.newaxis]
    dot = np.sum(q_vec * v, axis=-1, keepdims=True)
    c = q_vec * (2.0 * dot)

    return a - b + c


@dataclass
class XRTwistRetargeterCfg(RetargeterCfg):
    """Configuration for XRoboToolkit TWIST policy retargeter.

    This retargeter uses the TWIST policy to perform full-body motion tracking,
    combining GMR motion reference with proprioceptive robot state to output
    joint position targets.
    """

    twist_config_path: str = ""
    """Path to TWIST YAML configuration file containing model path, DOF indices, scales, etc."""

    robot_type: str = "booster_t1_29dof"
    """Target robot type for GMR retargeting. Supported: booster_t1_29dof, unitree_g1, etc."""

    human_height: float | None = None
    """Human height in meters. If None, auto-estimate from first frame."""

    use_ground_alignment: bool = True
    """If True, automatically align body to ground plane (important for headset-relative tracking)."""

    use_threading: bool = True
    """If True, run TWIST inference in a background thread for non-blocking operation."""

    thread_rate_hz: float = 50.0
    """Update rate for background inference thread in Hz (when use_threading=True)."""

    output_format: TwistOutputFormat = TwistOutputFormat.ABSOLUTE
    """Output format: ABSOLUTE (target positions) or OFFSET (position offsets)."""

    gmr_headless: bool = True
    """If True, run GMR without MuJoCo viewer visualization."""


class XRTwistRetargeter(RetargeterBase):
    """Retargets XR full-body tracking to robot using TWIST policy.

    This retargeter combines:
    1. XRGMRRetargeter - converts XR body tracking to motion reference
    2. Robot state sync - receives proprioceptive state from simulation
    3. TWIST ONNX policy - outputs joint position targets

    The TWIST policy learns to track motion references while maintaining balance,
    outputting joint position offsets from a default standing pose.

    Features:
    - Full-body motion tracking via GMR
    - Proprioceptive feedback from simulation state
    - Observation history buffer for temporal context
    - Threaded inference for non-blocking operation
    - Configurable output format (absolute or offset)
    """

    def __init__(self, cfg: XRTwistRetargeterCfg):
        """Initialize the TWIST retargeter.

        Args:
            cfg: Configuration for the retargeter

        Raises:
            RuntimeError: If ONNX runtime or rate limiter not available
            FileNotFoundError: If TWIST config file not found
        """
        super().__init__(cfg)

        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX runtime not available. Install with: pip install onnxruntime")

        if cfg.use_threading and not RATE_LIMITER_AVAILABLE:
            raise RuntimeError(
                "Threading mode requires loop_rate_limiters package. "
                "Install with: pip install loop-rate-limiters\n"
                "Or disable threading: use_threading=False"
            )

        # Store configuration
        self._robot_type = cfg.robot_type
        self._use_ground_alignment = cfg.use_ground_alignment
        self._output_format = cfg.output_format
        self._human_height = cfg.human_height
        self._use_threading = cfg.use_threading
        self._thread_rate_hz = cfg.thread_rate_hz
        self._gmr_headless = cfg.gmr_headless

        # Load TWIST configuration
        self._load_twist_config(cfg.twist_config_path)

        # Initialize GMR retargeter for motion reference
        gmr_cfg = XRGMRRetargeterCfg(
            sim_device=cfg.sim_device,
            robot_type=self._robot_type,
            human_height=self._human_height,
            use_ground_alignment=self._use_ground_alignment,
            output_format=GMROutputFormat.FULL_QPOS,
            headless=self._gmr_headless,
            use_threading=True,  # GMR always threaded for performance
            thread_rate_hz=90.0,
        )
        self._gmr_retargeter = XRGMRRetargeter(gmr_cfg)

        # Initialize ONNX policy
        self._load_onnx_model()

        # Initialize observation history buffer
        self._obs_history_buf = deque(maxlen=self._hist_len)
        for _ in range(self._hist_len):
            self._obs_history_buf.append(np.zeros(self._dim_single_obs))

        # Robot state from simulation (synced externally)
        self._robot_state = {
            "joint_pos": np.zeros(self._num_dofs),
            "joint_vel": np.zeros(self._num_dofs),
            "base_quat": np.array([0, 0, 0, 1]),  # [x, y, z, w]
            "base_ang_vel": np.zeros(3),
        }

        # Policy state
        self._last_action = np.zeros(self._n_policy_dofs)
        self._last_valid_output = None

        # Threading infrastructure
        if self._use_threading:
            self._datalock = threading.RLock()
            self._is_running = False
            self._is_ready = False
            self._input_xr_data = {}
            self._output_action = None
            self._inference_thread = None

        print(f"[XRTwistRetargeter] Initialized for robot: {self._robot_type}")
        print(f"[XRTwistRetargeter] Policy DOFs: {self._n_policy_dofs}")
        print(f"[XRTwistRetargeter] History length: {self._hist_len}")
        print(f"[XRTwistRetargeter] Output format: {self._output_format.value}")
        print(f"[XRTwistRetargeter] Threading: {self._use_threading}")

    def __del__(self):
        """Destructor to clean up thread resources."""
        if not hasattr(self, '_use_threading') or not self._use_threading:
            return
        if hasattr(self, '_is_running'):
            self._is_running = False
            if hasattr(self, '_inference_thread') and self._inference_thread is not None and self._inference_thread.is_alive():
                print("[XRTwistRetargeter] Stopping background thread...")
                self._inference_thread.join(timeout=2.0)
                if self._inference_thread.is_alive():
                    print("[XRTwistRetargeter] Warning: Thread did not stop cleanly")
                else:
                    print("[XRTwistRetargeter] Thread stopped")

    def _load_twist_config(self, config_path: str):
        """Load TWIST configuration from YAML file.

        Args:
            config_path: Path to TWIST YAML config file

        Raises:
            FileNotFoundError: If config file not found
            KeyError: If required config keys are missing
        """
        import os

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Store config directory for resolving relative paths
        config_dir = os.path.dirname(os.path.abspath(config_path))

        # Extract TWIST-specific configuration
        self._n_policy_dofs = config["TWIST_POLICY_DOFS"]
        self._hist_len = config["TWIST_HIST_LEN"]
        self._policy_dof_indices = config["TWIST_POLICY_INDICES"]
        self._ankle_indices = config.get("TWIST_ANKLE_INDICES", [])
        self._default_root_pos = np.array(config["TWIST_DEFAULT_ROOT_POS"])
        self._policy_action_scale = config.get("POLICY_ACTION_SCALE", 0.5)

        # Resolve model path - if relative, make it relative to config file location
        model_path = config["model_path"]
        if not os.path.isabs(model_path):
            # Try relative to config dir first, then relative to cwd
            model_path_from_config = os.path.join(config_dir, model_path)
            if os.path.exists(model_path_from_config):
                model_path = model_path_from_config
            # Otherwise keep original (relative to cwd)
        self._model_path = model_path

        # TWIST_DEFAULT_JOINT_POS contains default positions for policy DOFs only (27 values)
        # These correspond directly to the policy DOF ordering, not indexed by robot DOF index
        self._default_policy_joint_pos = np.array(config["TWIST_DEFAULT_JOINT_POS"])

        # Total robot DOFs (including head which policy doesn't control)
        # T1 has 29 DOFs: 2 head + 27 policy DOFs
        self._num_dofs = config.get("NUM_JOINTS", len(self._policy_dof_indices) + 2)

        # Build full robot default joint positions (29 DOFs)
        # Head DOFs (indices 0, 1) default to [0, 0]
        self._default_joint_pos = np.zeros(self._num_dofs)
        self._default_joint_pos[self._policy_dof_indices] = self._default_policy_joint_pos

        # Observation scales
        self._obs_scales = {
            "dof_pos": config.get("obs_scales", {}).get("dof_pos", 1.0),
            "dof_vel": config.get("obs_scales", {}).get("dof_vel", 0.05),
            "ang_vel": config.get("obs_scales", {}).get("ang_vel", 0.25),
        }

        # Calculate observation dimensions
        # Mimic obs: 1 (height) + 3 (euler) + 3 (linvel) + 1 (yaw_vel) + n_policy_dofs
        self._n_mimic_obs = 8 + self._n_policy_dofs
        # Proprio obs: 3 (ang_vel) + 2 (roll/pitch) + 3*n_policy_dofs (pos, vel, last_action)
        self._n_proprio_obs = 3 + 2 + 3 * self._n_policy_dofs
        self._dim_single_obs = self._n_mimic_obs + self._n_proprio_obs

        print(f"[XRTwistRetargeter] Loaded config from: {config_path}")
        print(f"[XRTwistRetargeter] Model path: {self._model_path}")
        print(f"[XRTwistRetargeter] Single obs dim: {self._dim_single_obs}")

    def _load_onnx_model(self):
        """Load ONNX policy model."""
        self._onnx_session = onnxruntime.InferenceSession(self._model_path)
        self._onnx_input_names = [inp.name for inp in self._onnx_session.get_inputs()]
        self._onnx_output_names = [out.name for out in self._onnx_session.get_outputs()]

        print(f"[XRTwistRetargeter] ONNX model loaded")
        print(f"[XRTwistRetargeter] Input names: {self._onnx_input_names}")

    def set_robot_state(self, state: dict[str, np.ndarray]):
        """Set robot state from simulation for proprioceptive observations.

        This method should be called each step with current robot state from
        the simulation environment.

        Args:
            state: Dictionary containing:
                - joint_pos: (n_dofs,) current joint positions
                - joint_vel: (n_dofs,) current joint velocities
                - base_quat: (4,) base orientation quaternion [x, y, z, w]
                - base_ang_vel: (3,) base angular velocity
        """
        if self._use_threading:
            with self._datalock:
                self._robot_state = {
                    "joint_pos": state["joint_pos"].copy(),
                    "joint_vel": state["joint_vel"].copy(),
                    "base_quat": state["base_quat"].copy(),
                    "base_ang_vel": state["base_ang_vel"].copy(),
                }
        else:
            self._robot_state = {
                "joint_pos": state["joint_pos"].copy(),
                "joint_vel": state["joint_vel"].copy(),
                "base_quat": state["base_quat"].copy(),
                "base_ang_vel": state["base_ang_vel"].copy(),
            }

    def reset(self, robot_state: dict[str, np.ndarray] | None = None):
        """Reset retargeter state.

        Clears observation history and optionally initializes with robot state.

        Args:
            robot_state: Optional initial robot state dictionary
        """
        # Reset observation history
        self._obs_history_buf.clear()
        for _ in range(self._hist_len):
            self._obs_history_buf.append(np.zeros(self._dim_single_obs))

        # Reset action history
        self._last_action = np.zeros(self._n_policy_dofs)
        self._last_valid_output = None

        # Set initial robot state if provided
        if robot_state is not None:
            self.set_robot_state(robot_state)

        # Reset GMR retargeter
        self._gmr_retargeter.reset()

        # Clear threading state
        if self._use_threading:
            with self._datalock:
                self._output_action = None
                self._is_ready = False

        print("[XRTwistRetargeter] Reset complete")

    def _get_mimic_obs(self, gmr_qpos: np.ndarray) -> np.ndarray:
        """Extract mimic observations from GMR output.

        Args:
            gmr_qpos: Full robot state from GMR [root_pos(3), root_quat(4), joint_angles(...)]

        Returns:
            Mimic observation vector (n_mimic_obs,)
        """
        # Extract components from GMR output
        root_pos = gmr_qpos[:3]
        root_quat_wxyz = gmr_qpos[3:7]  # GMR outputs [w, x, y, z]
        dof_pos = gmr_qpos[7:]

        # Convert quaternion to [w, x, y, z] for euler conversion (already in this format)
        roll, pitch, yaw = euler_from_quat(root_quat_wxyz)

        # Convert quaternion to [x, y, z, w] for velocity rotation
        root_quat_xyzw = root_quat_wxyz[[1, 2, 3, 0]]

        # Compute velocities from GMR (approximate - GMR doesn't provide velocities directly)
        # For now, use zeros - in practice you might compute finite differences
        root_linvel = np.zeros(3)
        root_angvel = np.zeros(3)

        # Transform velocities to body frame
        root_linvel_body = quat_rotate_inverse(root_quat_xyzw.reshape(1, -1), root_linvel.reshape(1, -1))[0]
        root_angvel_body = quat_rotate_inverse(root_quat_xyzw.reshape(1, -1), root_angvel.reshape(1, -1))[0]

        # Extract policy DOFs from GMR output
        dof_pos_policy = dof_pos[self._policy_dof_indices] if len(dof_pos) > max(self._policy_dof_indices) else np.zeros(self._n_policy_dofs)

        # Construct mimic observation
        mimic_obs = np.concatenate([
            root_pos[2:3],           # 1: height (z)
            [roll, pitch, yaw],      # 3: euler angles
            root_linvel_body,        # 3: linear velocity (body frame)
            root_angvel_body[2:3],   # 1: yaw angular velocity
            dof_pos_policy,          # n_policy_dofs: joint positions
        ])

        return mimic_obs

    def _get_proprio_obs(self) -> np.ndarray:
        """Extract proprioceptive observations from robot state.

        Returns:
            Proprioceptive observation vector (n_proprio_obs,)
        """
        # Get robot state (thread-safe if threading enabled)
        if self._use_threading:
            with self._datalock:
                joint_pos = self._robot_state["joint_pos"].copy()
                joint_vel = self._robot_state["joint_vel"].copy()
                base_quat = self._robot_state["base_quat"].copy()
                base_ang_vel = self._robot_state["base_ang_vel"].copy()
        else:
            joint_pos = self._robot_state["joint_pos"]
            joint_vel = self._robot_state["joint_vel"]
            base_quat = self._robot_state["base_quat"]
            base_ang_vel = self._robot_state["base_ang_vel"]

        # Convert quaternion [x, y, z, w] to [w, x, y, z] for euler
        base_quat_wxyz = base_quat[[3, 0, 1, 2]]
        roll, pitch, yaw = euler_from_quat(base_quat_wxyz)

        # Scale angular velocity
        ang_vel_scaled = base_ang_vel * self._obs_scales["ang_vel"]

        # Get policy DOF positions (relative to default)
        dof_pos = joint_pos[self._policy_dof_indices] - self._default_joint_pos[self._policy_dof_indices]
        dof_pos_scaled = dof_pos * self._obs_scales["dof_pos"]

        # Get policy DOF velocities
        dof_vel = joint_vel[self._policy_dof_indices].copy()
        # Zero out ankle velocities (as in TWIST)
        for idx in self._ankle_indices:
            if idx in self._policy_dof_indices:
                local_idx = self._policy_dof_indices.index(idx)
                dof_vel[local_idx] = 0.0
        dof_vel_scaled = dof_vel * self._obs_scales["dof_vel"] * 0.01

        # Construct proprioceptive observation
        proprio_obs = np.concatenate([
            ang_vel_scaled,          # 3: angular velocity
            [roll, pitch],           # 2: IMU orientation
            dof_pos_scaled,          # n_policy_dofs: joint positions
            dof_vel_scaled,          # n_policy_dofs: joint velocities
            self._last_action,       # n_policy_dofs: previous action
        ])

        return proprio_obs

    def _construct_obs(self, gmr_qpos: np.ndarray) -> np.ndarray:
        """Construct full observation with history.

        Args:
            gmr_qpos: GMR output for mimic observations

        Returns:
            Full observation vector with history (1, total_obs_dim)
        """
        # Get current observations
        mimic_obs = self._get_mimic_obs(gmr_qpos)
        proprio_obs = self._get_proprio_obs()
        current_obs = np.concatenate([mimic_obs, proprio_obs])

        # Get history and append current
        obs_hist = np.array(self._obs_history_buf)
        obs_full = np.concatenate([current_obs, obs_hist.flatten()])

        # Update history buffer
        self._obs_history_buf.append(current_obs)

        return obs_full.reshape(1, -1)

    def _policy_inference(self, obs: np.ndarray) -> np.ndarray:
        """Run ONNX policy inference.

        Args:
            obs: Full observation array (1, total_obs_dim)

        Returns:
            Policy action (n_policy_dofs,)
        """
        input_feed = {self._onnx_input_names[0]: obs.astype(np.float32)}
        outputs = self._onnx_session.run(self._onnx_output_names, input_feed)
        action = outputs[0][0]  # (1, n_policy_dofs) -> (n_policy_dofs,)
        return action

    def _start_inference_thread(self):
        """Start the background inference thread."""
        if not self._use_threading:
            return

        self._is_running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="TwistInferenceThread",
            daemon=True
        )
        self._inference_thread.start()
        print(f"[XRTwistRetargeter] Background thread started at {self._thread_rate_hz} Hz")

    def _inference_loop(self):
        """Main loop for background inference thread."""
        rate = RateLimiter(self._thread_rate_hz)

        while self._is_running:
            try:
                # Read input XR data (protected by lock)
                with self._datalock:
                    if not self._input_xr_data:
                        self._is_ready = False
                        rate.sleep()
                        continue
                    xr_data = self._input_xr_data.copy()

                # Get GMR motion reference (non-blocking call to GMR)
                gmr_output = self._gmr_retargeter.retarget(xr_data)
                if gmr_output is None:
                    rate.sleep()
                    continue

                gmr_qpos = gmr_output.cpu().numpy()

                # Construct observation and run inference
                obs = self._construct_obs(gmr_qpos)
                action = self._policy_inference(obs)

                # Scale and store action
                scaled_action = action * self._policy_action_scale
                self._last_action = action.copy()

                # Format output
                if self._output_format == TwistOutputFormat.ABSOLUTE:
                    output = np.zeros(self._num_dofs)
                    output[self._policy_dof_indices] = scaled_action + self._default_joint_pos[self._policy_dof_indices]
                else:
                    output = scaled_action

                # Write output (protected by lock)
                with self._datalock:
                    self._output_action = output
                    self._is_ready = True

            except Exception as e:
                print(f"[TwistInferenceThread] Error: {e}")
                import traceback
                traceback.print_exc()

            rate.sleep()

        print("[TwistInferenceThread] Thread stopped")

    def retarget(self, data: dict[str, Any]) -> torch.Tensor | None:
        """Convert XR full-body tracking data to robot joint positions.

        In threaded mode: Updates input buffer and reads latest output (non-blocking).
        In non-threaded mode: Performs synchronous inference (blocking).

        Args:
            data: Dictionary from XRControllerFullBodyDevice containing:
                - 'body_joints': {joint_name: [x, y, z, qx, qy, qz, qw]} (24 joints)
                - 'buttons': button states
                - 'timestamp': timestamp

        Returns:
            torch.Tensor | None: Joint positions or offsets based on output_format,
                or None if not ready yet
        """
        if self._use_threading:
            return self._retarget_threaded(data)
        else:
            return self._retarget_synchronous(data)

    def _retarget_threaded(self, data: dict[str, Any]) -> torch.Tensor | None:
        """Non-blocking retargeting using background thread.

        Args:
            data: XR device data dictionary

        Returns:
            torch.Tensor | None: Latest retargeted output or cached output
        """
        # Start thread on first call
        if self._inference_thread is None:
            self._start_inference_thread()

        # Update input buffer (thread-safe)
        with self._datalock:
            self._input_xr_data = data

            # Read latest output
            if self._output_action is not None and self._is_ready:
                output = self._output_action.copy()
            else:
                return self._last_valid_output

        # Convert to torch tensor
        output_tensor = torch.tensor(output, dtype=torch.float32, device=self._sim_device)
        self._last_valid_output = output_tensor

        return output_tensor

    def _retarget_synchronous(self, data: dict[str, Any]) -> torch.Tensor | None:
        """Synchronous (blocking) retargeting.

        Args:
            data: XR device data dictionary

        Returns:
            torch.Tensor | None: Retargeted output or None on error
        """
        try:
            # Get GMR motion reference
            gmr_output = self._gmr_retargeter.retarget(data)
            if gmr_output is None:
                return self._last_valid_output

            gmr_qpos = gmr_output.cpu().numpy()

            # Construct observation and run inference
            obs = self._construct_obs(gmr_qpos)
            action = self._policy_inference(obs)

            # Scale and store action
            scaled_action = action * self._policy_action_scale
            self._last_action = action.copy()

            # Format output
            if self._output_format == TwistOutputFormat.ABSOLUTE:
                output = np.zeros(self._num_dofs)
                output[self._policy_dof_indices] = scaled_action + self._default_joint_pos[self._policy_dof_indices]
            else:
                output = scaled_action

            # Convert to torch tensor
            output_tensor = torch.tensor(output, dtype=torch.float32, device=self._sim_device)
            self._last_valid_output = output_tensor

            return output_tensor

        except Exception as e:
            print(f"[XRTwistRetargeter] Error: {e}")
            import traceback
            traceback.print_exc()
            return self._last_valid_output
