"""Policy Provider for LeRobot policies in Isaac Lab.

This module bridges LeRobot pretrained policies with Isaac Lab environments,
handling observation format conversion and action processing. It provides a
clean interface for loading and running any LeRobot policy type (diffusion,
ACT, VQ-BeT, flow matching, etc.) in Isaac Lab simulations.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.converters import transition_to_policy_action
from lerobot.processor.core import EnvTransition, PolicyAction, TransitionKey
from scripts.tools.lerobot_action_chunk_broker import ActionChunkBroker


def policy_action_observation_to_transition(
    action_observation: tuple[PolicyAction, dict[str, Any]],
) -> EnvTransition:
    """
    Convert a policy action tensor and observation dictionary into an EnvTransition.

    This custom converter allows the postprocessor to accept both action and observation,
    enabling AbsoluteJointActionsProcessor to access the observation for delta->absolute conversion.

    Args:
        action_observation: Tuple of (action tensor, observation dict)

    Returns:
        EnvTransition containing the action and observation
    """
    if not isinstance(action_observation, tuple):
        raise ValueError("action_observation should be a tuple type with an action and observation")

    action, observation = action_observation

    if action is not None and not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

    if observation is not None and not isinstance(observation, dict):
        raise ValueError(f"Observation should be a dict type got {type(observation)}")

    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: None,
        TransitionKey.DONE: None,
        TransitionKey.TRUNCATED: None,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }


class LeRobotPolicyProvider:
    """
    Provider class that loads and serves LeRobot policies for Isaac Lab environments.

    This class handles:
    - Loading pretrained policies from HuggingFace Hub or local paths
    - Converting Isaac Lab observations to LeRobot format
    - Converting LeRobot actions to Isaac Lab format
    - Optional action chunking to reduce inference frequency
    - Policy state management (reset, device placement)

    The provider is policy-agnostic and works with any LeRobot policy type through
    the PreTrainedPolicy interface.

    Args:
        model_path: Path to pretrained policy (HuggingFace Hub ID or local path)
        device: Device for inference ("cuda", "cpu", "mps")
        use_action_chunking: Whether to use action chunking to reduce inference frequency
        execution_horizon: Number of steps before triggering new inference (if action chunking enabled)
        image_keys: Single image key or list of image keys from Isaac Lab observations
                   (default: ["head_rgb_cam"]). Maps to LeRobot observation keys like:
                   "head_rgb_cam" -> "observation.images.head_rgb"
                   "left_wrist_cam" -> "observation.images.left_wrist"
                   "right_wrist_cam" -> "observation.images.right_wrist"
        image_key_mapping: Explicit mapping from Isaac Lab obs key to LeRobot observation key.
                          Example: {"head_cam_rgb": "observation.images.head_rgb"}
                          If None, falls back to heuristic: replace("_cam", "") then prefix.
        state_key: Name of the state observation in Isaac Lab (default: "joint_pos").
                  Used when state_keys is not provided.
        state_keys: List of state observation keys to concatenate. Overrides state_key when provided.
                   Example: ["robot_joint_pos", "hand_joint_state"]
        upper_body_dof: Number of upper body DOFs in action output. If None, auto-detected from policy config.
        state_dof: Number of state DOFs to pass to policy. If None, auto-detected from policy config.

    Example:
        >>> # Single camera (T1 defaults)
        >>> provider = LeRobotPolicyProvider(
        ...     model_path="kelvinzhaozg/t1_stack_cube_policy",
        ...     device="cuda",
        ...     use_action_chunking=True,
        ...     execution_horizon=8
        ... )
        >>> # G1 with explicit key mapping and auto-detect dimensions
        >>> provider = LeRobotPolicyProvider(
        ...     model_path="path/to/g1_policy",
        ...     device="cuda",
        ...     image_keys=["head_cam_rgb"],
        ...     image_key_mapping={"head_cam_rgb": "observation.images.head_rgb"},
        ...     state_key="robot_joint_pos",
        ...     upper_body_dof=None,  # auto-detect from policy config
        ...     state_dof=None,       # auto-detect from policy config
        ... )
        >>> provider.reset()
        >>> obs = env.observation_manager.compute_group("policy")
        >>> action = provider.get_action(obs)
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        use_action_chunking: bool = True,
        execution_horizon: int = 8,
        image_keys: str | list[str] = "head_rgb_cam",
        image_key_mapping: dict[str, str] | None = None,
        state_key: str = "joint_pos",
        state_keys: list[str] | None = None,
        upper_body_dof: int | None = None,
        state_dof: int | None = None,
    ):
        """
        Initialize the policy provider.

        Args:
            model_path: Path to pretrained policy (hub ID or local path)
            device: Device for inference ("cuda", "cpu", "mps")
            use_action_chunking: Whether to use action chunking
            execution_horizon: Number of steps before starting next inference
            image_keys: Single image key or list of image keys for Isaac Lab observations
            image_key_mapping: Explicit mapping from Isaac Lab obs key to LeRobot observation key.
                             If None, falls back to heuristic (replace "_cam" suffix).
            state_key: Key for joint positions in Isaac Lab observations (used if state_keys is None)
            state_keys: List of state obs keys to concatenate. Overrides state_key when provided.
            upper_body_dof: Number of upper body DOFs (actions). None = auto-detect from policy config.
            state_dof: Number of state DOFs for policy input. None = auto-detect from policy config.
        """
        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.use_action_chunking = use_action_chunking
        self.execution_horizon = execution_horizon

        # Convert image_keys to list if single string provided
        if isinstance(image_keys, str):
            self.image_keys = [image_keys]
        else:
            self.image_keys = image_keys

        self.image_key_mapping = image_key_mapping
        self.state_key = state_key
        self.state_keys = state_keys

        # Store user-provided values (may be None for auto-detect)
        self._user_upper_body_dof = upper_body_dof
        self._user_state_dof = state_dof

        # Load the policy using LeRobot's from_pretrained
        self._load_policy()

        # Auto-detect dimensions from policy config if not explicitly provided
        self._resolve_dimensions()

        # Setup action chunking if enabled
        if self.use_action_chunking:
            self.action_broker = ActionChunkBroker(
                policy=self.policy,
                execution_horizon=self.execution_horizon,
            )
            self.inference_policy = self.action_broker
        else:
            self.inference_policy = self.policy

        print(f"LeRobotPolicyProvider initialized:")
        print(f"  Model: {self.model_path}")
        print(f"  Policy type: {self.policy.config.type}")
        print(f"  Device: {device}")
        print(f"  Action chunking: {use_action_chunking}")
        if use_action_chunking:
            print(f"  Execution horizon: {execution_horizon}")
        print(f"  Image observations: {', '.join(self.image_keys)}")
        if self.image_key_mapping:
            print(f"  Image key mapping: {self.image_key_mapping}")
        if self.state_keys:
            print(f"  State keys: {self.state_keys}")
        else:
            print(f"  State key: {self.state_key}")
        print(f"  State input dim: {self.state_dof}")
        print(f"  Action output dim: {self.upper_body_dof}")

    def _load_policy(self):
        """Load the pretrained policy using LeRobot's factory."""
        try:
            print(f"Loading policy from: {self.model_path}")

            # First, load the config to determine policy type
            config = PreTrainedConfig.from_pretrained(self.model_path)
            print(f"  Detected policy type: {config.type}")

            # Get the concrete policy class for this type
            policy_cls = get_policy_class(config.type)

            # Load the pretrained policy
            self.policy = policy_cls.from_pretrained(self.model_path)
            self.policy.to(self.device)
            self.policy.eval()
            print(f"✓ Loaded {config.type} policy successfully")

            # Override n_action_steps with execution_horizon when not using action chunking
            if not self.use_action_chunking and hasattr(config, 'n_action_steps'):
                original_n_action_steps = config.n_action_steps
                config.n_action_steps = self.execution_horizon

                # Validate constraint for policies with horizon (e.g., diffusion, flow matching)
                if hasattr(config, 'horizon') and hasattr(config, 'n_obs_steps'):
                    max_allowed = config.horizon - config.n_obs_steps + 1
                    if config.n_action_steps > max_allowed:
                        raise ValueError(
                            f"execution_horizon ({self.execution_horizon}) exceeds maximum allowed "
                            f"({max_allowed}) for this policy. Maximum is: "
                            f"horizon ({config.horizon}) - n_obs_steps ({config.n_obs_steps}) + 1"
                        )

                print(f"  Overriding n_action_steps: {original_n_action_steps} -> {config.n_action_steps}")

            # Load the preprocessor and postprocessor pipelines
            # These handle normalization/unnormalization automatically
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=config,
                pretrained_path=self.model_path,
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
                postprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )

            # Override postprocessor's to_transition to accept (action, observation) tuple
            # This allows AbsoluteJointActionsProcessor to access unnormalized observation
            # for delta->absolute conversion: absolute_action = delta_action + current_state
            self.postprocessor.to_transition = policy_action_observation_to_transition
            self.postprocessor.to_output = transition_to_policy_action

            print(f"✓ Loaded preprocessor and postprocessor pipelines")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load policy from {self.model_path}. "
                f"Make sure the path is a valid HuggingFace Hub ID or local directory. "
                f"Error: {e}"
            )

    def _resolve_dimensions(self):
        """Resolve upper_body_dof and state_dof from user values or policy config auto-detection."""
        config = self.policy.config

        # Print full feature information for debugging
        print("  Policy config features:")
        if hasattr(config, "input_features"):
            for key, feat in config.input_features.items():
                print(f"    input:  {key} -> shape={feat.shape}, type={feat.type}")
        if hasattr(config, "output_features"):
            for key, feat in config.output_features.items():
                print(f"    output: {key} -> shape={feat.shape}, type={feat.type}")

        # Auto-detect action dimension from policy config
        auto_action_dim = None
        action_ft = getattr(config, "action_feature", None)
        if action_ft is not None and hasattr(action_ft, "shape"):
            auto_action_dim = action_ft.shape[0]

        # Auto-detect state dimension from policy config
        auto_state_dim = None
        state_ft = getattr(config, "robot_state_feature", None)
        if state_ft is not None and hasattr(state_ft, "shape"):
            auto_state_dim = state_ft.shape[0]

        # Resolve upper_body_dof
        if self._user_upper_body_dof is not None:
            self.upper_body_dof = self._user_upper_body_dof
            if auto_action_dim is not None and auto_action_dim != self._user_upper_body_dof:
                print(f"  Note: user override upper_body_dof={self._user_upper_body_dof}, "
                      f"policy config has action_dim={auto_action_dim}")
        elif auto_action_dim is not None:
            self.upper_body_dof = auto_action_dim
            print(f"  Auto-detected upper_body_dof={auto_action_dim} from policy config")
        else:
            self.upper_body_dof = 18  # Legacy default (T1)
            print(f"  Warning: could not auto-detect action dim, using default upper_body_dof=18")

        # Resolve state_dof
        if self._user_state_dof is not None:
            self.state_dof = self._user_state_dof
            if auto_state_dim is not None and auto_state_dim != self._user_state_dof:
                print(f"  Note: user override state_dof={self._user_state_dof}, "
                      f"policy config has state_dim={auto_state_dim}")
        elif auto_state_dim is not None:
            self.state_dof = auto_state_dim
            print(f"  Auto-detected state_dof={auto_state_dim} from policy config")
        else:
            self.state_dof = 21  # Legacy default (T1)
            print(f"  Warning: could not auto-detect state dim, using default state_dof=21")

        # Validate image key mapping against policy config's expected image features
        image_features = getattr(config, "image_features", None)
        if image_features:
            expected_image_keys = set(image_features.keys())
            if self.image_key_mapping:
                mapped_keys = set(self.image_key_mapping.values())
                missing = expected_image_keys - mapped_keys
                extra = mapped_keys - expected_image_keys
                if missing:
                    print(f"  WARNING: Policy expects image keys not in mapping: {missing}")
                if extra:
                    print(f"  WARNING: Mapping provides image keys not expected by policy: {extra}")
            else:
                # Show what the policy expects so user can verify heuristic mapping
                print(f"  Policy expects image keys: {list(expected_image_keys)}")
                # Show what the heuristic would produce
                for image_key in self.image_keys:
                    camera_name = image_key.replace("_cam", "")
                    heuristic_key = f"observation.images.{camera_name}"
                    if heuristic_key not in expected_image_keys:
                        print(f"  WARNING: Heuristic maps '{image_key}' -> '{heuristic_key}' "
                              f"but policy expects {list(expected_image_keys)}")

    def reset(self):
        """Reset the policy and action broker state.

        This should be called whenever the environment is reset to clear
        cached observations/actions and prepare for a new episode.
        """
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        if hasattr(self, "action_broker"):
            self.action_broker.reset()

    def preprocess_observation(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Convert observation from Isaac Lab format to LeRobot format.

        Isaac Lab provides observations through the observation manager:
        - obs[image_key]: (num_envs, height, width, channels) uint8
        - obs[state_key]: (num_envs, num_joints) float32

        LeRobot expects:
        - "observation.images.{camera_name}": (batch, channels, height, width) float32 [0, 1]
        - "observation.state": (batch, state_dim) float32

        Args:
            obs: Observation dictionary from Isaac Lab observation manager
                Expected keys: {image_keys: Tensor, state_key: Tensor}

        Returns:
            Preprocessed observation ready for policy inference with LeRobot format
        """
        # First, convert Isaac Lab format to LeRobot raw format
        raw_obs = {}

        # Handle RGB images (potentially multiple cameras)
        for image_key in self.image_keys:
            if image_key in obs:
                rgb_image = obs[image_key]

                # Extract first environment (we only support single env evaluation for now)
                if len(rgb_image.shape) == 4:
                    # (num_envs, H, W, C) -> (H, W, C)
                    rgb_image = rgb_image[0]

                # Convert to torch tensor if numpy
                if isinstance(rgb_image, np.ndarray):
                    rgb_image = torch.from_numpy(rgb_image)

                # Normalize to [0, 1] if uint8
                if rgb_image.dtype == torch.uint8:
                    rgb_image = rgb_image.float() / 255.0

                # Convert from (H, W, C) to (C, H, W) - channels first WITHOUT batch yet
                if len(rgb_image.shape) == 3:
                    rgb_image = rgb_image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

                # Map Isaac Lab camera names to LeRobot observation keys
                if self.image_key_mapping and image_key in self.image_key_mapping:
                    lerobot_key = self.image_key_mapping[image_key]
                else:
                    # Fallback heuristic: "head_rgb_cam" -> "head_rgb" -> "observation.images.head_rgb"
                    camera_name = image_key.replace("_cam", "")
                    lerobot_key = f"observation.images.{camera_name}"
                raw_obs[lerobot_key] = rgb_image
            else:
                raise ValueError(
                    f"Expected image key '{image_key}' not found in observations. "
                    f"Available keys: {list(obs.keys())}"
                )

        # Handle state observation
        # Determine which keys to use for state
        keys_for_state = self.state_keys if self.state_keys else [self.state_key]

        state_parts = []
        for s_key in keys_for_state:
            if s_key in obs:
                s = obs[s_key]

                # Extract first environment
                if len(s.shape) == 2:
                    s = s[0]

                # Convert to torch tensor if numpy
                if isinstance(s, np.ndarray):
                    s = torch.from_numpy(s).float()

                state_parts.append(s)
            else:
                raise ValueError(
                    f"Expected state key '{s_key}' not found in observations. "
                    f"Available keys: {list(obs.keys())}"
                )

        # Concatenate state parts if multiple keys
        if len(state_parts) == 1:
            state = state_parts[0]
        else:
            state = torch.cat(state_parts, dim=-1)

        # Extract state DOFs for policy (first state_dof elements)
        state = state[: self.state_dof]

        raw_obs["observation.state"] = state

        # Apply preprocessing pipeline (adds batch dim, normalizes, etc.)
        processed_obs = self.preprocessor(raw_obs)

        # Prepare unnormalized observation for delta->absolute action conversion
        # The AbsoluteJointActionsProcessor needs unnormalized state with batch dimension
        raw_obs_with_batch = {}
        for key, value in raw_obs.items():
            # Add batch dimension to match processed_obs format
            if isinstance(value, torch.Tensor):
                raw_obs_with_batch[key] = value.unsqueeze(0)  # (dim,) -> (1, dim) or (C, H, W) -> (1, C, H, W)
            else:
                raw_obs_with_batch[key] = value

        # Move to same device as processed_obs
        for key, value in raw_obs_with_batch.items():
            if isinstance(value, torch.Tensor):
                raw_obs_with_batch[key] = value.to(self.device)

        return processed_obs, raw_obs_with_batch

    def get_action(self, obs: dict[str, Tensor]) -> Tensor:
        """
        Generate action from current observation.

        This is the main interface for getting actions from the policy. It handles:
        1. Preprocessing observations to LeRobot format
        2. Running inference (with action chunking if enabled)
        3. Returning actions as torch tensor for Isaac Lab

        Args:
            obs: Observation dictionary from Isaac Lab
                Should contain keys: {image_key, state_key}

        Returns:
            Action as torch tensor of shape (upper_body_dof,)
            For T1 robot, this is (18,) joint positions
        """
        # Preprocess observation (normalize, add batch dim, etc.)
        # Returns both processed (normalized) and raw (unnormalized with batch dim) observations
        processed_obs, raw_obs_with_batch = self.preprocess_observation(obs)

        # Generate action using policy (potentially through action broker)
        with torch.no_grad():
            if self.use_action_chunking:
                action = self.action_broker.infer(processed_obs)
            else:
                action = self.policy.select_action(processed_obs)

        # Postprocess action (unnormalize, optionally convert delta->absolute)
        action = self.postprocessor((action, raw_obs_with_batch))

        # Ensure it's a tensor
        if not isinstance(action, Tensor):
            action = torch.from_numpy(action).to(self.device)

        # Remove batch dimension if present
        if len(action.shape) == 2 and action.shape[0] == 1:
            action = action.squeeze(0)

        # Validate action shape
        if action.shape[-1] != self.upper_body_dof:
            raise ValueError(
                f"Policy output has wrong action dimension: {action.shape[-1]}. "
                f"Expected {self.upper_body_dof} for upper body DOFs"
            )

        return action

    def validate_observation_format(self, obs: dict[str, Tensor]) -> bool:
        """
        Validate that observation has the expected format.

        Args:
            obs: Observation to validate

        Returns:
            True if format is valid, False otherwise (with warnings printed)
        """
        state_keys_to_check = self.state_keys if self.state_keys else [self.state_key]
        required_keys = self.image_keys + state_keys_to_check

        for key in required_keys:
            if key not in obs:
                print(f"❌ Missing observation key: {key}")
                print(f"   Available keys: {list(obs.keys())}")
                return False

        # Check image format for all cameras
        for image_key in self.image_keys:
            rgb = obs[image_key]
            if len(rgb.shape) != 4 or rgb.shape[-1] != 3:
                print(f"❌ Invalid RGB image shape for {image_key}: {rgb.shape}")
                print(f"   Expected: (num_envs, height, width, 3)")
                return False

        # Check state format - sum dimensions across all state keys
        total_state_dim = 0
        for s_key in state_keys_to_check:
            state = obs[s_key]
            if len(state.shape) != 2:
                print(f"❌ Invalid state shape for '{s_key}': {state.shape}")
                print(f"   Expected: (num_envs, num_joints)")
                return False
            total_state_dim += state.shape[-1]

        if total_state_dim < self.state_dof:
            print(f"❌ Total state dimension too small: {total_state_dim}")
            print(f"   Expected at least {self.state_dof} joints for policy input")
            return False

        return True

    @property
    def policy_info(self) -> dict[str, Any]:
        """Get information about the loaded policy for debugging."""
        info = {
            "model_path": self.model_path,
            "policy_type": self.policy.config.type,
            "device": str(self.device),
            "use_action_chunking": self.use_action_chunking,
            "state_dim": self.state_dof,
            "action_dim": self.upper_body_dof,
            "image_keys": self.image_keys,
            "image_key_mapping": self.image_key_mapping,
            "state_key": self.state_key,
            "state_keys": self.state_keys,
        }

        if self.use_action_chunking:
            info.update(
                {
                    "execution_horizon": self.execution_horizon,
                    "chunk_progress": self.action_broker.cache_progress
                    if hasattr(self, "action_broker")
                    else (0, 0),
                }
            )

        # Add policy-specific config if available
        if hasattr(self.policy, "config"):
            info["policy_config"] = {
                "horizon": getattr(self.policy.config, "horizon", None),
                "n_obs_steps": getattr(self.policy.config, "n_obs_steps", None),
                "n_action_steps": getattr(self.policy.config, "n_action_steps", None),
            }

        return info
