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
from scripts.tools.lerobot_action_chunk_broker import ActionChunkBroker


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
        image_key: Name of the image observation in Isaac Lab (default: "head_rgb_cam")
        state_key: Name of the state observation in Isaac Lab (default: "joint_pos")
        upper_body_dof: Number of upper body DOFs in action output (default: 18 for T1)
        state_dof: Number of state DOFs to pass to policy (default: 21 for all T1 joints)

    Example:
        >>> provider = LeRobotPolicyProvider(
        ...     model_path="kelvinzhaozg/t1_stack_cube_policy",
        ...     device="cuda",
        ...     use_action_chunking=True,
        ...     execution_horizon=8
        ... )
        >>> provider.reset()
        >>> obs = env.observation_manager.compute_group("policy")
        >>> action = provider.get_action(obs)  # Returns (18,) joint positions
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        use_action_chunking: bool = True,
        execution_horizon: int = 8,
        image_key: str = "head_rgb_cam",
        state_key: str = "joint_pos",
        upper_body_dof: int = 18,
        state_dof: int = 21,
    ):
        """
        Initialize the policy provider.

        Args:
            model_path: Path to pretrained policy (hub ID or local path)
            device: Device for inference ("cuda", "cpu", "mps")
            use_action_chunking: Whether to use action chunking
            execution_horizon: Number of steps before starting next inference
            image_key: Key for RGB image in Isaac Lab observations
            state_key: Key for joint positions in Isaac Lab observations
            upper_body_dof: Number of upper body DOFs (actions will be this size)
            state_dof: Number of state DOFs to pass to policy (default: 21 for all T1 joints)
        """
        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.use_action_chunking = use_action_chunking
        self.execution_horizon = execution_horizon
        self.image_key = image_key
        self.state_key = state_key
        self.upper_body_dof = upper_body_dof
        self.state_dof = state_dof

        # Load the policy using LeRobot's from_pretrained
        self._load_policy()

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
        print(f"  State input dim: {state_dof}")
        print(f"  Action output dim: {upper_body_dof}")

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

            # Load the preprocessor and postprocessor pipelines
            # These handle normalization/unnormalization automatically
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=config,
                pretrained_path=self.model_path,
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
                postprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )
            print(f"✓ Loaded preprocessor and postprocessor pipelines")

            # Monkey-patch policy.unnormalize_outputs to use the postprocessor
            # DiffusionPolicy internally calls self.unnormalize_outputs() during select_action()
            # We redirect this to use our postprocessor pipeline
            def unnormalize_outputs(output_dict):
                """Unnormalize outputs using the postprocessor pipeline."""
                if "action" in output_dict:
                    unnormalized = self.postprocessor(output_dict["action"])
                    return {"action": unnormalized}
                return output_dict

            self.policy.unnormalize_outputs = unnormalize_outputs

        except Exception as e:
            raise RuntimeError(
                f"Failed to load policy from {self.model_path}. "
                f"Make sure the path is a valid HuggingFace Hub ID or local directory. "
                f"Error: {e}"
            )

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
        - "observation.images.head_rgb": (batch, channels, height, width) float32 [0, 1]
        - "observation.state": (batch, state_dim) float32

        Args:
            obs: Observation dictionary from Isaac Lab observation manager
                Expected keys: {image_key: Tensor, state_key: Tensor}

        Returns:
            Preprocessed observation ready for policy inference with LeRobot format
        """
        # First, convert Isaac Lab format to LeRobot raw format
        raw_obs = {}

        # Handle RGB image
        if self.image_key in obs:
            rgb_image = obs[self.image_key]

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

            raw_obs["observation.images.head_rgb"] = rgb_image
        else:
            raise ValueError(
                f"Expected image key '{self.image_key}' not found in observations. "
                f"Available keys: {list(obs.keys())}"
            )

        # Handle state observation
        if self.state_key in obs:
            state = obs[self.state_key]

            # Extract first environment
            if len(state.shape) == 2:
                # (num_envs, state_dim) -> (state_dim,)
                state = state[0]

            # Convert to torch tensor if numpy
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()

            # Extract state DOFs for policy (first state_dof elements)
            state = state[: self.state_dof]

            raw_obs["observation.state"] = state
        else:
            raise ValueError(
                f"Expected state key '{self.state_key}' not found in observations. "
                f"Available keys: {list(obs.keys())}"
            )

        # Apply preprocessing pipeline (adds batch dim, normalizes, etc.)
        processed_obs = self.preprocessor(raw_obs)
        return processed_obs

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
        # Preprocess observation
        processed_obs = self.preprocess_observation(obs)

        # Generate action using policy (potentially through action broker)
        # Note: Unnormalization happens internally via policy.unnormalize_outputs()
        with torch.no_grad():
            if self.use_action_chunking:
                action = self.action_broker.infer(processed_obs)
            else:
                action = self.policy.select_action(processed_obs)

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
        required_keys = [self.image_key, self.state_key]

        for key in required_keys:
            if key not in obs:
                print(f"❌ Missing observation key: {key}")
                print(f"   Available keys: {list(obs.keys())}")
                return False

        # Check image format
        rgb = obs[self.image_key]
        if len(rgb.shape) != 4 or rgb.shape[-1] != 3:
            print(f"❌ Invalid RGB image shape: {rgb.shape}")
            print(f"   Expected: (num_envs, height, width, 3)")
            return False

        # Check state format
        state = obs[self.state_key]
        if len(state.shape) != 2:
            print(f"❌ Invalid state shape: {state.shape}")
            print(f"   Expected: (num_envs, num_joints)")
            return False

        if state.shape[-1] < self.state_dof:
            print(f"❌ State dimension too small: {state.shape[-1]}")
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
            "image_key": self.image_key,
            "state_key": self.state_key,
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
