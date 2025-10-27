#!/usr/bin/env python3
"""
Simple test script to validate LeRobot policy integration with Isaac Lab.

This script performs basic validation of the policy provider without requiring
a trained policy or running a full simulation.
"""

import sys
from pathlib import Path

import numpy as np
import torch


def test_action_chunk_broker():
    """Test the action chunk broker in isolation."""
    print("\n" + "=" * 70)
    print("TEST 1: Action Chunk Broker")
    print("=" * 70)

    from scripts.tools.lerobot_action_chunk_broker import ActionChunkBroker

    # Create a mock policy that generates dummy action chunks
    class MockPolicy:
        def __init__(self):
            self.call_count = 0

        def predict_action_chunk(self, observations):
            self.call_count += 1
            # Return dummy action chunk: (1, 32, 18)
            return torch.randn(1, 32, 18)

        def reset(self):
            pass

    # Test basic functionality
    policy = MockPolicy()
    broker = ActionChunkBroker(policy, execution_horizon=8)

    print(f"Initial state:")
    print(f"  Is caching: {broker.is_caching}")
    print(f"  Cache progress: {broker.cache_progress}")

    # Mock observation
    obs = {
        "observation.images.head_rgb": torch.randn(1, 3, 240, 424),
        "observation.state": torch.randn(1, 18),
    }

    # First inference should trigger policy call
    action1 = broker.infer(obs)
    print(f"\nAfter first inference:")
    print(f"  Action shape: {action1.shape}")
    print(f"  Policy calls: {policy.call_count}")
    print(f"  Is caching: {broker.is_caching}")
    print(f"  Cache progress: {broker.cache_progress}")

    assert action1.shape == (18,), f"Expected action shape (18,), got {action1.shape}"
    assert policy.call_count == 1, f"Expected 1 policy call, got {policy.call_count}"

    # Next 7 calls should use cache
    for i in range(7):
        action = broker.infer(obs)
        assert action.shape == (18,)

    print(f"\nAfter 7 more inferences (using cache):")
    print(f"  Policy calls: {policy.call_count}")
    print(f"  Cache progress: {broker.cache_progress}")

    assert policy.call_count == 1, f"Expected still 1 policy call, got {policy.call_count}"

    # 8th call should trigger new inference
    action9 = broker.infer(obs)
    print(f"\nAfter 9th inference (should re-infer):")
    print(f"  Policy calls: {policy.call_count}")
    print(f"  Cache progress: {broker.cache_progress}")

    assert policy.call_count == 2, f"Expected 2 policy calls, got {policy.call_count}"

    # Test reset
    broker.reset()
    print(f"\nAfter reset:")
    print(f"  Is caching: {broker.is_caching}")
    print(f"  Cache progress: {broker.cache_progress}")

    assert not broker.is_caching, "Broker should not be caching after reset"

    print("\n✓ Action chunk broker test PASSED")
    return True


def test_policy_provider_preprocessing():
    """Test observation preprocessing without loading a real policy."""
    print("\n" + "=" * 70)
    print("TEST 2: Policy Provider Observation Preprocessing")
    print("=" * 70)

    # Create mock Isaac Lab observations
    isaac_obs = {
        "head_rgb_cam": torch.randint(0, 255, (1, 240, 424, 3), dtype=torch.uint8),  # (num_envs, H, W, C)
        "joint_pos": torch.randn(1, 21),  # (num_envs, 21 joints)
    }

    print(f"Isaac Lab observation format:")
    print(f"  head_rgb_cam: {isaac_obs['head_rgb_cam'].shape} {isaac_obs['head_rgb_cam'].dtype}")
    print(f"  joint_pos: {isaac_obs['joint_pos'].shape} {isaac_obs['joint_pos'].dtype}")

    # We can't fully test PolicyProvider without a real policy,
    # but we can test the preprocessing logic in isolation
    from scripts.tools.lerobot_policy_provider import LeRobotPolicyProvider

    # Create a minimal mock for testing preprocessing
    class MockPreprocessor:
        def __init__(self):
            self.device = torch.device("cpu")
            self.image_key = "head_rgb_cam"
            self.state_key = "joint_pos"
            self.upper_body_dof = 18

        def preprocess_observation(self, obs):
            # Copy the actual preprocessing logic
            processed_obs = {}

            if self.image_key in obs:
                rgb_image = obs[self.image_key]
                if len(rgb_image.shape) == 4:
                    rgb_image = rgb_image[0]
                if rgb_image.dtype == torch.uint8:
                    rgb_image = rgb_image.float() / 255.0
                if len(rgb_image.shape) == 3:
                    rgb_image = rgb_image.permute(2, 0, 1).unsqueeze(0)
                processed_obs["observation.images.head_rgb"] = rgb_image.to(self.device)

            if self.state_key in obs:
                state = obs[self.state_key]
                if len(state.shape) == 2:
                    state = state[0]
                state = state[: self.upper_body_dof]
                if len(state.shape) == 1:
                    state = state.unsqueeze(0)
                processed_obs["observation.state"] = state.to(self.device)

            return processed_obs

    preprocessor = MockPreprocessor()
    lerobot_obs = preprocessor.preprocess_observation(isaac_obs)

    print(f"\nLeRobot observation format:")
    print(f"  observation.images.head_rgb: {lerobot_obs['observation.images.head_rgb'].shape}")
    print(f"  observation.state: {lerobot_obs['observation.state'].shape}")

    # Validate shapes
    assert lerobot_obs["observation.images.head_rgb"].shape == (1, 3, 240, 424), (
        f"Expected image shape (1, 3, 240, 424), "
        f"got {lerobot_obs['observation.images.head_rgb'].shape}"
    )

    assert lerobot_obs["observation.state"].shape == (1, 18), (
        f"Expected state shape (1, 18), " f"got {lerobot_obs['observation.state'].shape}"
    )

    # Validate image normalization
    img = lerobot_obs["observation.images.head_rgb"]
    assert img.dtype == torch.float32, f"Expected float32, got {img.dtype}"
    assert img.min() >= 0.0 and img.max() <= 1.0, f"Image not in [0, 1] range: [{img.min()}, {img.max()}]"

    print("\n✓ Observation preprocessing test PASSED")
    return True


def test_imports():
    """Test that all required imports work."""
    print("\n" + "=" * 70)
    print("TEST 0: Import Validation")
    print("=" * 70)

    try:
        from scripts.tools.lerobot_action_chunk_broker import ActionChunkBroker

        print("✓ ActionChunkBroker imported successfully")

        from scripts.tools.lerobot_policy_provider import LeRobotPolicyProvider

        print("✓ LeRobotPolicyProvider imported successfully")

        print("\n✓ All imports PASSED")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("LeRobot Integration Test Suite")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Action Chunk Broker", test_action_chunk_broker),
        ("Observation Preprocessing", test_policy_provider_preprocessing),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' FAILED with exception:")
            print(f"  {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(r for _, r in results)
    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
