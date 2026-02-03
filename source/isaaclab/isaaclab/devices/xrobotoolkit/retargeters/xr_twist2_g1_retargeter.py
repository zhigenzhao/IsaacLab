# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit controller retargeter using TWIST2 policy for G1 humanoid full-body control."""

import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# Default path to TWIST2 model (relative to this module)
_DEFAULT_TWIST2_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "twist2", "twist2_1017_20k.onnx"
)

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg

from .xr_gmr_retargeter import GMROutputFormat, XRGMRRetargeter, XRGMRRetargeterCfg
from .xr_twist_retargeter import TwistOutputFormat, euler_from_quat, quat_rotate_inverse

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
    print("Warning: ONNX runtime not available. XRTwist2G1Retargeter will not function.")


# G1 29-DOF joint ordering for TWIST2 policy
# This matches the joint order in TWIST2 training
G1_TWIST2_JOINT_NAMES = [
    # Left leg (6 DOFs)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6 DOFs)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Torso (3 DOFs)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (7 DOFs)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (7 DOFs)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Number of DOFs for TWIST2 G1 policy
NUM_DOFS = 29

# Ankle joint indices (velocities zeroed in observation)
ANKLE_INDICES = [4, 5, 10, 11]  # left_ankle_pitch, left_ankle_roll, right_ankle_pitch, right_ankle_roll

# Default standing pose for G1 (29 DOFs)
DEFAULT_DOF_POS = np.array([
    # Left leg (6)
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    # Right leg (6)
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    # Torso (3)
    0.0, 0.0, 0.0,
    # Left arm (7)
    0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
    # Right arm (7)
    0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
])

# TWIST2 observation structure (from server_low_level_g1_sim.py):
# mimic_obs = xy_vel(2) + z(1) + roll(1) + pitch(1) + yaw_vel(1) + dof_pos(29) = 35
# proprio_obs = ang_vel*0.25(3) + rpy[:2](2) + dof_pos_offset(29) + dof_vel*0.05(29) + last_action(29) = 92
# obs_single = mimic_obs + proprio_obs = 35 + 92 = 127
# obs_hist = 10 frames * 127 = 1270
# future_obs = mimic_obs = 35
# total = obs_single * (history_len + 1) + mimic_obs = 127 * 11 + 35 = 1432
DIM_MIMIC_OBS = 35
DIM_PROPRIO_OBS = 92
DIM_SINGLE_OBS = DIM_MIMIC_OBS + DIM_PROPRIO_OBS  # 127
HISTORY_LENGTH = 10
DIM_FUTURE_OBS = DIM_MIMIC_OBS  # 35 - future trajectory uses same dims as mimic_obs


class XRTwist2G1Retargeter(RetargeterBase):
    """Retargets XR full-body tracking to G1 robot using TWIST2 policy.

    This retargeter combines:
    1. XRGMRRetargeter - converts XR body tracking to motion reference for G1
    2. Robot state sync - receives proprioceptive state from simulation
    3. TWIST2 ONNX policy - outputs joint position PD targets (29 DOFs)

    The TWIST2 policy learns to track motion references while maintaining balance,
    outputting joint position offsets from a default standing pose.

    Features:
    - Full-body motion tracking via GMR (unitree_g1 robot)
    - Proprioceptive feedback from simulation state
    - Observation history buffer for temporal context (10 frames)
    - Threaded inference for non-blocking operation (50Hz)
    - Configurable output format (absolute or offset)
    """

    def __init__(self, cfg: "XRTwist2G1RetargeterCfg"):
        """Initialize the TWIST2 G1 retargeter.

        Args:
            cfg: Configuration for the retargeter

        Raises:
            RuntimeError: If ONNX runtime or rate limiter not available
            FileNotFoundError: If TWIST2 ONNX model not found
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
        self._action_scale = cfg.action_scale
        self._history_length = cfg.history_length
        self._policy_path = cfg.policy_path

        # Initialize GMR retargeter for motion reference
        gmr_cfg = XRGMRRetargeterCfg(
            sim_device=cfg.sim_device,
            robot_type=self._robot_type,
            human_height=self._human_height,
            use_ground_alignment=self._use_ground_alignment,
            output_format=GMROutputFormat.FULL_QPOS,
            headless=self._gmr_headless,
            use_threading=True,  # GMR always threaded for performance
            thread_rate_hz=60.0,
        )
        self._gmr_retargeter = XRGMRRetargeter(gmr_cfg)

        # Initialize ONNX policy
        self._load_onnx_model()

        # Initialize observation history buffer
        self._obs_history_buf = deque(maxlen=self._history_length)
        for _ in range(self._history_length):
            self._obs_history_buf.append(np.zeros(DIM_SINGLE_OBS))

        # Robot state from simulation (synced externally)
        self._robot_state = {
            "joint_pos": np.zeros(NUM_DOFS),
            "joint_vel": np.zeros(NUM_DOFS),
            "base_quat": np.array([0, 0, 0, 1]),  # [x, y, z, w]
            "base_ang_vel": np.zeros(3),
        }

        # GMR state tracking for velocity computation
        self._last_gmr_qpos = None
        self._last_gmr_time = None

        # Policy state
        self._last_action = np.zeros(NUM_DOFS)
        self._last_valid_output = None

        # Threading infrastructure
        if self._use_threading:
            self._datalock = threading.RLock()
            self._is_running = False
            self._is_ready = False
            self._input_xr_data = {}
            self._output_action = None
            self._inference_thread = None

        print(f"[XRTwist2G1Retargeter] Initialized for robot: {self._robot_type}")
        print(f"[XRTwist2G1Retargeter] Policy path: {self._policy_path}")
        print(f"[XRTwist2G1Retargeter] Policy DOFs: {NUM_DOFS}")
        print(f"[XRTwist2G1Retargeter] History length: {self._history_length}")
        print(f"[XRTwist2G1Retargeter] Action scale: {self._action_scale}")
        print(f"[XRTwist2G1Retargeter] Output format: {self._output_format.value}")
        print(f"[XRTwist2G1Retargeter] Threading: {self._use_threading}")

    def __del__(self):
        """Destructor to clean up thread resources."""
        if not hasattr(self, '_use_threading') or not self._use_threading:
            return
        if hasattr(self, '_is_running'):
            self._is_running = False
            if hasattr(self, '_inference_thread') and self._inference_thread is not None and self._inference_thread.is_alive():
                print("[XRTwist2G1Retargeter] Stopping background thread...")
                self._inference_thread.join(timeout=2.0)
                if self._inference_thread.is_alive():
                    print("[XRTwist2G1Retargeter] Warning: Thread did not stop cleanly")
                else:
                    print("[XRTwist2G1Retargeter] Thread stopped")

    def _load_onnx_model(self):
        """Load ONNX policy model."""
        import os
        if not os.path.exists(self._policy_path):
            raise FileNotFoundError(f"TWIST2 ONNX model not found at: {self._policy_path}")

        self._onnx_session = onnxruntime.InferenceSession(self._policy_path)
        self._onnx_input_names = [inp.name for inp in self._onnx_session.get_inputs()]
        self._onnx_output_names = [out.name for out in self._onnx_session.get_outputs()]

        # Log model info
        input_shape = self._onnx_session.get_inputs()[0].shape
        output_shape = self._onnx_session.get_outputs()[0].shape
        print(f"[XRTwist2G1Retargeter] ONNX model loaded")
        print(f"[XRTwist2G1Retargeter] Input names: {self._onnx_input_names}")
        print(f"[XRTwist2G1Retargeter] Input shape: {input_shape}")
        print(f"[XRTwist2G1Retargeter] Output shape: {output_shape}")

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
        for _ in range(self._history_length):
            self._obs_history_buf.append(np.zeros(DIM_SINGLE_OBS))

        # Reset action history
        self._last_action = np.zeros(NUM_DOFS)
        self._last_valid_output = None

        # Reset GMR tracking
        self._last_gmr_qpos = None
        self._last_gmr_time = None

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

        print("[XRTwist2G1Retargeter] Reset complete")

    def _get_mimic_obs(self, gmr_qpos: np.ndarray) -> np.ndarray:
        """Extract mimic observations from GMR output.

        TWIST2 mimic observation (35 dims):
        - xy_vel (2): root linear velocity in body frame [x, y]
        - z (1): root height
        - roll (1): roll angle
        - pitch (1): pitch angle
        - yaw_vel (1): yaw angular velocity
        - dof_pos (29): joint positions

        Args:
            gmr_qpos: Full robot state from GMR [root_pos(3), root_quat(4), joint_angles(29)]

        Returns:
            Mimic observation vector (35,)
        """
        import time as time_module

        # Extract components from GMR output
        root_pos = gmr_qpos[:3]
        root_quat_wxyz = gmr_qpos[3:7]  # GMR outputs [w, x, y, z]
        dof_pos = gmr_qpos[7:]  # 29 DOFs for G1

        # Convert quaternion to euler angles
        roll, pitch, yaw = euler_from_quat(root_quat_wxyz)

        # Convert quaternion to [x, y, z, w] for velocity rotation
        root_quat_xyzw = root_quat_wxyz[[1, 2, 3, 0]]

        # Compute velocities from GMR via finite difference
        current_time = time_module.time()
        if self._last_gmr_qpos is not None and self._last_gmr_time is not None:
            dt = current_time - self._last_gmr_time
            if dt > 0.001:  # Avoid division by very small dt
                root_linvel = (root_pos - self._last_gmr_qpos[:3]) / dt
                # Compute yaw velocity from quaternion difference
                last_roll, last_pitch, last_yaw = euler_from_quat(self._last_gmr_qpos[3:7])
                yaw_vel = (yaw - last_yaw) / dt
                # Handle wraparound
                if yaw_vel > np.pi:
                    yaw_vel -= 2 * np.pi
                elif yaw_vel < -np.pi:
                    yaw_vel += 2 * np.pi
            else:
                root_linvel = np.zeros(3)
                yaw_vel = 0.0
        else:
            root_linvel = np.zeros(3)
            yaw_vel = 0.0

        # Update tracking state
        self._last_gmr_qpos = gmr_qpos.copy()
        self._last_gmr_time = current_time

        # Transform linear velocity to body frame
        root_linvel_body = quat_rotate_inverse(root_quat_xyzw.reshape(1, -1), root_linvel.reshape(1, -1))[0]

        # Construct mimic observation (35 dims)
        mimic_obs = np.concatenate([
            root_linvel_body[:2],     # 2: xy velocity (body frame)
            root_pos[2:3],            # 1: height (z)
            [roll],                   # 1: roll
            [pitch],                  # 1: pitch
            [yaw_vel],                # 1: yaw angular velocity
            dof_pos[:NUM_DOFS],       # 29: joint positions
        ])

        return mimic_obs

    def _get_proprio_obs(self) -> np.ndarray:
        """Extract proprioceptive observations from robot state.

        TWIST2 proprio observation (92 dims):
        - ang_vel * 0.25 (3): scaled angular velocity
        - rpy[:2] (2): roll, pitch
        - dof_pos_offset (29): joint positions relative to default
        - dof_vel * 0.05 (29): scaled joint velocities
        - last_action (29): previous action

        Returns:
            Proprioceptive observation vector (92,)
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
        ang_vel_scaled = base_ang_vel * 0.25

        # Get DOF positions (relative to default)
        dof_pos_offset = joint_pos[:NUM_DOFS] - DEFAULT_DOF_POS

        # Get DOF velocities and zero out ankles
        dof_vel = joint_vel[:NUM_DOFS].copy()
        for idx in ANKLE_INDICES:
            dof_vel[idx] = 0.0
        dof_vel_scaled = dof_vel * 0.05

        # Construct proprioceptive observation (92 dims)
        proprio_obs = np.concatenate([
            ang_vel_scaled,          # 3: scaled angular velocity
            [roll, pitch],           # 2: roll, pitch
            dof_pos_offset,          # 29: joint positions offset
            dof_vel_scaled,          # 29: scaled joint velocities
            self._last_action,       # 29: previous action
        ])

        return proprio_obs

    def _construct_obs(self, gmr_qpos: np.ndarray) -> np.ndarray:
        """Construct full observation with history.

        TWIST2 observation structure (1432 dims total):
        - obs_full (127): mimic_obs(35) + obs_proprio(92)
        - obs_hist (1270): 10 frames * 127
        - future_obs (35): future trajectory (uses current mimic_obs)

        Args:
            gmr_qpos: GMR output for mimic observations

        Returns:
            Full observation vector with history (1, 1432)
        """
        # Get current observations
        mimic_obs = self._get_mimic_obs(gmr_qpos)
        proprio_obs = self._get_proprio_obs()
        current_obs = np.concatenate([mimic_obs, proprio_obs])

        # Get history (oldest to newest, flattened)
        obs_hist = np.array(self._obs_history_buf).flatten()

        # Future observation - use current mimic_obs as placeholder
        # (matches TWIST2 deployment: future_obs = action_mimic.copy())
        future_obs = mimic_obs.copy()

        # Full observation
        obs_full = np.concatenate([current_obs, obs_hist, future_obs])

        # Update history buffer (append current observation)
        self._obs_history_buf.append(current_obs)

        return obs_full.reshape(1, -1)

    def _policy_inference(self, obs: np.ndarray) -> np.ndarray:
        """Run ONNX policy inference.

        Args:
            obs: Full observation array (1, total_obs_dim)

        Returns:
            Policy action (NUM_DOFS,)
        """
        input_feed = {self._onnx_input_names[0]: obs.astype(np.float32)}
        outputs = self._onnx_session.run(self._onnx_output_names, input_feed)
        action = outputs[0][0]  # (1, NUM_DOFS) -> (NUM_DOFS,)
        return action

    def _process_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Process raw policy action to PD targets.

        Args:
            raw_action: Raw policy output (NUM_DOFS,)

        Returns:
            PD target positions (NUM_DOFS,)
        """
        # Clip raw action
        raw_action = np.clip(raw_action, -10.0, 10.0)

        # Scale action
        scaled_action = raw_action * self._action_scale

        # Add to default pose to get absolute position targets
        pd_target = scaled_action + DEFAULT_DOF_POS

        return pd_target

    def _start_inference_thread(self):
        """Start the background inference thread."""
        if not self._use_threading:
            return

        self._is_running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="Twist2G1InferenceThread",
            daemon=True
        )
        self._inference_thread.start()
        print(f"[XRTwist2G1Retargeter] Background thread started at {self._thread_rate_hz} Hz")

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

                # Process action to PD targets
                pd_target = self._process_action(action)
                self._last_action = action.copy()

                # Format output
                if self._output_format == TwistOutputFormat.ABSOLUTE:
                    output = pd_target
                else:
                    output = action * self._action_scale  # OFFSET format

                # Write output (protected by lock)
                with self._datalock:
                    self._output_action = output
                    self._is_ready = True

            except Exception as e:
                print(f"[Twist2G1InferenceThread] Error: {e}")
                import traceback
                traceback.print_exc()

            rate.sleep()

        print("[Twist2G1InferenceThread] Thread stopped")

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
            torch.Tensor | None: Joint position PD targets (29 dims) or None if not ready
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

            # Process action to PD targets
            pd_target = self._process_action(action)
            self._last_action = action.copy()

            # Format output
            if self._output_format == TwistOutputFormat.ABSOLUTE:
                output = pd_target
            else:
                output = action * self._action_scale

            # Convert to torch tensor
            output_tensor = torch.tensor(output, dtype=torch.float32, device=self._sim_device)
            self._last_valid_output = output_tensor

            return output_tensor

        except Exception as e:
            print(f"[XRTwist2G1Retargeter] Error: {e}")
            import traceback
            traceback.print_exc()
            return self._last_valid_output


@dataclass
class XRTwist2G1RetargeterCfg(RetargeterCfg):
    """Configuration for XRoboToolkit TWIST2 G1 policy retargeter.

    This retargeter uses the TWIST2 policy to perform full-body motion tracking
    on the Unitree G1 humanoid, combining GMR motion reference with proprioceptive
    robot state to output joint position PD targets (29 DOFs).
    """

    policy_path: str = _DEFAULT_TWIST2_MODEL_PATH
    """Path to TWIST2 ONNX policy model."""

    robot_type: str = "unitree_g1"
    """Target robot type for GMR retargeting. Must be unitree_g1 for this retargeter."""

    human_height: float | None = None
    """Human height in meters. If None, auto-estimate from first frame."""

    use_ground_alignment: bool = True
    """If True, automatically align body to ground plane (important for headset-relative tracking)."""

    gmr_headless: bool = True
    """If True, run GMR without MuJoCo viewer visualization."""

    action_scale: float = 0.5
    """Scale factor for policy action output."""

    history_length: int = 10
    """Number of observation history frames for temporal context."""

    use_threading: bool = True
    """If True, run TWIST2 inference in a background thread for non-blocking operation."""

    thread_rate_hz: float = 50.0
    """Update rate for background inference thread in Hz (when use_threading=True)."""

    output_format: TwistOutputFormat = TwistOutputFormat.ABSOLUTE
    """Output format: ABSOLUTE (PD target positions) or OFFSET (position offsets)."""

    retargeter_type: type[RetargeterBase] = XRTwist2G1Retargeter
