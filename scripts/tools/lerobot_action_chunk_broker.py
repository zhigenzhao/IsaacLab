"""Action chunk broker for LeRobot pretrained policies in Isaac Lab.

This module provides a simplified action chunking mechanism adapted from the thirdarm project
for use with Isaac Lab environments. It caches action sequences from policies and returns them
sequentially to reduce inference frequency.
"""

from typing import Optional

import torch
from torch import Tensor


class ActionChunkBroker:
    """
    A policy wrapper that returns action chunks sequentially from a LeRobot policy.

    This broker manages action chunks across an action horizon, returning one action
    at a time from the policy's complete action sequence. It automatically triggers
    new policy inference when chunks are exhausted.

    Args:
        policy: The underlying LeRobot PreTrainedPolicy instance
        execution_horizon: Number of actions to use before requesting new chunk.
            If None, uses all actions from the chunk.

    Example:
        >>> from lerobot.policies.pretrained import PreTrainedPolicy
        >>> policy = PreTrainedPolicy.from_pretrained("path/to/model")
        >>> broker = ActionChunkBroker(policy, execution_horizon=8)
        >>>
        >>> # First call triggers inference
        >>> action = broker.infer(observations)  # Gets action 0 from new chunk
        >>> # Subsequent calls use cached chunk
        >>> action = broker.infer(observations)  # Gets action 1 from cache
        >>> # ... 6 more calls use cache
        >>> action = broker.infer(observations)  # Gets action 8, triggers new inference
    """

    def __init__(self, policy, execution_horizon: Optional[int] = None):
        """
        Initialize the action chunk broker.

        Args:
            policy: The underlying LeRobot PreTrainedPolicy
            execution_horizon: Number of actions to use before requesting new chunk.
                If None, uses all actions in the chunk (no early re-inference).
        """
        self.policy = policy
        self.execution_horizon = execution_horizon
        self.last_results: Optional[Tensor] = None
        self.current_step: int = 0

    def infer(self, observations: dict[str, Tensor]) -> Tensor:
        """
        Infer the next action from the policy.

        This method manages the action cache and triggers new inference when needed.
        On the first call or when the cache is exhausted, it calls the policy to generate
        a new action chunk. Otherwise, it returns the next cached action.

        Args:
            observations: Dictionary of observations (images, states, etc.)
                Expected format matches LeRobot conventions:
                - "observation.images.head_rgb": (batch, channels, height, width)
                - "observation.state": (batch, state_dim)

        Returns:
            Tensor: Single action vector of shape (action_dim,)
                For Isaac Lab T1 tasks, this is typically (18,) for joint positions.
        """
        # Check if we need new predictions
        needs_inference = self.last_results is None
        if not needs_inference and self.execution_horizon is not None:
            needs_inference = self.current_step >= self.execution_horizon
        elif not needs_inference:
            # No execution horizon set, use all actions in chunk
            needs_inference = self.current_step >= self.last_results.shape[1]

        if needs_inference:
            # Call policy to get new action chunk
            # Expected output shape: (batch_size, time_steps, action_dim)
            self.last_results = self.policy.predict_action_chunk(observations)
            self.current_step = 0

        # Extract single action from sequence
        # Handle different possible tensor shapes from various policy types
        if len(self.last_results.shape) == 3:
            # Shape: (batch, time, action_dim)
            batch_size = self.last_results.shape[0]
            if batch_size == 1:
                # Remove batch dimension: (1, time, action_dim) -> (time, action_dim)
                actions_sequence = self.last_results[0]
            else:
                # Multiple batches - take first one
                actions_sequence = self.last_results[0]
            current_action = actions_sequence[self.current_step]
        elif len(self.last_results.shape) == 2:
            # Shape: (time, action_dim) - already no batch dimension
            current_action = self.last_results[self.current_step]
        else:
            raise ValueError(
                f"Unexpected action chunk shape: {self.last_results.shape}. "
                f"Expected (batch, time, action_dim) or (time, action_dim)"
            )

        self.current_step += 1
        return current_action

    def reset(self):
        """Reset the broker state.

        This should be called when the environment is reset to clear cached actions
        and force new inference on the next call.
        """
        self.last_results = None
        self.current_step = 0

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying policy.

        This allows the broker to be used as a drop-in replacement for the policy
        while adding action chunking functionality.
        """
        return getattr(self.policy, name)

    @property
    def is_caching(self) -> bool:
        """Check if the broker currently has cached actions."""
        return self.last_results is not None

    @property
    def cache_progress(self) -> tuple[int, int]:
        """Get current cache utilization.

        Returns:
            tuple[int, int]: (current_step, total_steps) in the cached chunk.
                Returns (0, 0) if no cache is active.
        """
        if self.last_results is None:
            return (0, 0)

        # Determine total steps based on execution_horizon or chunk size
        if self.execution_horizon is not None:
            total_steps = min(self.execution_horizon, self.last_results.shape[1])
        else:
            total_steps = self.last_results.shape[1]

        return (self.current_step, total_steps)
