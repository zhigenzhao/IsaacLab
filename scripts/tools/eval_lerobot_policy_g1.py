#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to evaluate LeRobot pretrained policies on G1 tasks in Isaac Lab environments.

This script loads a pretrained LeRobot policy (diffusion, ACT, VQ-BeT, flow matching, etc.)
and evaluates it in an Isaac Lab environment configured for the Unitree G1 robot with
Inspire hand. It leverages the generalized LeRobotPolicyProvider with auto-detection of
action/state dimensions from the policy config.

Usage:
    # Evaluate with head camera (default for XR variant)
    python scripts/tools/eval_lerobot_policy_g1.py \
        --task Isaac-PickPlace-G1-InspireFTP-XR-v0 \
        --policy_path path/to/g1_policy \
        --num_episodes 10 \
        --use_action_chunking \
        --policy_device cuda

    # Evaluate with video recording
    python scripts/tools/eval_lerobot_policy_g1.py \
        --task Isaac-PickPlace-G1-InspireFTP-XR-v0 \
        --policy_path path/to/g1_policy \
        --num_episodes 5 \
        --use_action_chunking \
        --record_video \
        --policy_device cuda

    # Evaluate a 16-dim compact action policy (14 arm + 2 gripper triggers)
    python scripts/tools/eval_lerobot_policy_g1.py \
        --task Isaac-PickPlace-G1-InspireFTP-XR-v0 \
        --policy_path path/to/compact_policy \
        --num_episodes 10 \
        --use_action_chunking \
        --compact_actions \
        --policy_device cuda
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

# Isaac Lab imports
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate LeRobot policy on G1 tasks in Isaac Lab environment.")
parser.add_argument(
    "--task", type=str, required=True,
    help="Name of the Isaac Lab G1 task (e.g. Isaac-PickPlace-G1-InspireFTP-XR-v0)."
)
parser.add_argument(
    "--policy_path", type=str, required=True, help="Path to pretrained LeRobot policy (HF Hub ID or local path)."
)
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate.")
parser.add_argument("--max_episode_length", type=int, default=500, help="Maximum steps per episode.")
parser.add_argument(
    "--use_action_chunking", action="store_true", default=False, help="Enable action chunking to reduce inference."
)
parser.add_argument("--execution_horizon", type=int, default=8, help="Steps before re-inference (if chunking enabled).")
parser.add_argument("--policy_device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Device for policy inference.")
parser.add_argument("--record_video", action="store_true", default=False, help="Record video of rollouts.")
parser.add_argument("--video_dir", type=str, default="./videos", help="Directory to save videos.")
parser.add_argument("--video_fps", type=int, default=30, help="FPS for recorded videos.")
parser.add_argument("--save_trajectories", action="store_true", default=False, help="Save trajectory data.")
parser.add_argument("--trajectory_dir", type=str, default="./trajectories", help="Directory to save trajectories.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument(
    "--compact_actions", action="store_true", default=False,
    help="Enable 16-dim action mode (14 arm + 2 gripper). Expands gripper commands to 24 hand joints via interpolation."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Automatically enable cameras if video recording is requested
if args_cli.record_video:
    args_cli.enable_cameras = True
    print("Video recording enabled - cameras will be rendered even in headless mode")

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys
from pathlib import Path

# Add Isaac Lab root to Python path for importing scripts.tools modules
ISAACLAB_ROOT = Path(__file__).resolve().parents[2]
if str(ISAACLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAACLAB_ROOT))

import gymnasium as gym
import isaaclab_tasks  # noqa: F401

# Explicitly import pick_place to trigger gym registration
# (pick_place is blacklisted from auto-import due to pinocchio compatibility)
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
import numpy as np
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from scripts.tools.lerobot_policy_provider import LeRobotPolicyProvider

# Hand joint expansion for compact_actions mode
from isaaclab.devices.xrobotoolkit.retargeters.xr_inspire_hand_retargeter import INSPIRE_HAND_JOINT_LIMITS
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_unitree_g1_inspire_hand_xr_env_cfg import (
    G1_HAND_JOINT_NAMES,
)


def expand_gripper_to_hand_joints(
    left_trigger: float, right_trigger: float, hand_joint_names: list[str], device: str = "cpu"
) -> torch.Tensor:
    """Expand 2 gripper trigger values [0-1] to 24 hand joint positions.

    Uses the same linear interpolation as XRInspireHandRetargeter:
        value = open_pos + trigger * (closed_pos - open_pos)

    Args:
        left_trigger: Left hand gripper command in [0, 1].
        right_trigger: Right hand gripper command in [0, 1].
        hand_joint_names: Ordered list of 24 hand joint names.
        device: Torch device for the output tensor.

    Returns:
        Tensor of shape (24,) with interpolated hand joint positions.
    """
    joint_values: list[float] = []
    for name in hand_joint_names:
        # Extract joint type by stripping L_/R_ prefix and _joint suffix
        jtype = name
        if jtype.startswith("L_") or jtype.startswith("R_"):
            jtype = jtype[2:]
        if jtype.endswith("_joint"):
            jtype = jtype[:-6]

        open_pos, closed_pos = INSPIRE_HAND_JOINT_LIMITS.get(jtype, (0.0, 1.7))
        trigger = left_trigger if name.startswith("L_") else right_trigger
        value = open_pos + trigger * (closed_pos - open_pos)
        joint_values.append(value)

    return torch.tensor(joint_values, dtype=torch.float32, device=device)


# Import video writer
try:
    import imageio
    VIDEO_BACKEND = "imageio"
except ImportError:
    try:
        import cv2
        VIDEO_BACKEND = "opencv"
    except ImportError:
        VIDEO_BACKEND = None


class VideoWriter:
    """Simple video writer that works in headless mode."""

    def __init__(self, video_path: Path, fps: int = 30):
        self.video_path = video_path
        self.fps = fps
        self.frames = []

    def add_frame(self, frame: np.ndarray):
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        self.frames.append(frame)

    def save(self):
        if not self.frames:
            print(f"Warning: No frames to save for {self.video_path}")
            return

        if VIDEO_BACKEND == "imageio":
            imageio.mimsave(self.video_path, self.frames, fps=self.fps)
        elif VIDEO_BACKEND == "opencv":
            height, width = self.frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (width, height))
            for frame in self.frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        else:
            print("Warning: No video backend available. Install imageio or opencv-python.")

        print(f"  Saved video: {self.video_path} ({len(self.frames)} frames)")


@dataclass
class EpisodeMetrics:
    """Container for episode-level metrics."""

    episode_num: int
    episode_length: int
    total_reward: float
    success: bool
    inference_time_avg: float
    inference_time_total: float


class PolicyEvaluator:
    """Evaluator for LeRobot policies on G1 tasks in Isaac Lab environments."""

    def __init__(self, args):
        self.args = args
        self.metrics_history: list[EpisodeMetrics] = []

        # Set random seeds for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Create output directories if needed
        if args.record_video:
            Path(args.video_dir).mkdir(parents=True, exist_ok=True)
            if VIDEO_BACKEND is None:
                print("WARNING: Video recording requested but no backend available!")
                print("  Install imageio: pip install imageio[ffmpeg]")
                print("  Or install opencv: pip install opencv-python")
                print("  Video recording will be disabled.")
                args.record_video = False
            else:
                print(f"Video backend: {VIDEO_BACKEND}")
        if args.save_trajectories:
            Path(args.trajectory_dir).mkdir(parents=True, exist_ok=True)

        # Initialize environment
        self._setup_environment()

        # Initialize policy provider
        self._setup_policy()

        print("\n" + "=" * 70)
        print("LeRobot Policy Evaluation - G1 Robot")
        print("=" * 70)
        print(f"Task: {args.task}")
        print(f"Policy: {args.policy_path}")
        print(f"Episodes: {args.num_episodes}")
        print(f"Max episode length: {args.max_episode_length}")
        print(f"Action chunking: {args.use_action_chunking}")
        if args.use_action_chunking:
            print(f"Execution horizon: {args.execution_horizon}")
        print(f"Compact actions (16-dim): {args.compact_actions}")
        print(f"Video recording: {args.record_video}")
        if args.record_video:
            print(f"  Output dir: {args.video_dir}")
            print(f"  FPS: {args.video_fps}")
        print(f"Headless mode: {args.headless}")
        print("=" * 70 + "\n")

    def _setup_environment(self):
        """Setup the Isaac Lab environment for G1."""
        print(f"Setting up environment: {self.args.task}")

        # Parse environment configuration
        try:
            env_cfg = parse_env_cfg(self.args.task, device=args_cli.device, num_envs=1)
        except Exception as e:
            print(f"Failed to parse environment configuration: {e}")
            raise

        # Track whether head camera is available in the scene
        self._has_head_cam = False

        if self.args.record_video or args_cli.enable_cameras:
            print("Adding camera observations to policy group...")
            import isaaclab.sim as sim_utils
            from isaaclab.envs import mdp
            from isaaclab.managers import ObservationTermCfg as ObsTerm
            from isaaclab.managers import SceneEntityCfg
            from isaaclab.sensors import CameraCfg

            # Enable DLSS antialiasing
            env_cfg.sim.render.antialiasing_mode = "DLSS"

            # Check if head_cam entity already exists in the scene config
            # (the XR variant defines it; non-XR variants may not)
            if hasattr(env_cfg.scene, "head_cam"):
                self._has_head_cam = True
                # Add head camera observation term if not already present
                if not hasattr(env_cfg.observations.policy, "head_cam_rgb"):
                    env_cfg.observations.policy.head_cam_rgb = ObsTerm(
                        func=mdp.image,
                        params={"sensor_cfg": SceneEntityCfg("head_cam"), "data_type": "rgb", "normalize": False}
                    )
                print("  head_cam found in scene config - adding head_cam_rgb observation")
            else:
                print("  Warning: head_cam not found in scene config. "
                      "Head camera recording will be skipped. "
                      "Use the XR variant (e.g. Isaac-PickPlace-G1-InspireFTP-XR-v0) for head camera support.")

            # Add scene camera (third-person view) for video recording
            env_cfg.scene.scene_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/scene_cam",
                update_period=0.0,
                height=480,
                width=640,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 20.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(2, 2, 1.25),
                    rot=(0.27, 0.27, 0.65, 0.65),
                    convention="opengl",
                ),
            )
            env_cfg.observations.policy.scene_cam = ObsTerm(
                func=mdp.image,
                params={"sensor_cfg": SceneEntityCfg("scene_cam"), "data_type": "rgb", "normalize": False}
            )

            cameras_added = ["scene"]
            if self._has_head_cam:
                cameras_added.insert(0, "head")
            print(f"  Camera observations added: {', '.join(cameras_added)}")

        # Disable time-out termination for evaluation
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None

        # Create environment
        try:
            self.env = gym.make(self.args.task, cfg=env_cfg).unwrapped
        except Exception as e:
            print(f"Failed to create environment: {e}")
            raise

        print(f"  Environment created successfully")
        print(f"  Observation space: {self.env.observation_space}")
        print(f"  Action space: {self.env.action_space}")

    def _setup_policy(self):
        """Setup the LeRobot policy provider for G1."""
        print(f"\nSetting up policy provider...")

        self._compact_actions = self.args.compact_actions

        # G1 image keys and mapping
        image_keys = []
        image_key_mapping = {}
        if self._has_head_cam:
            image_keys.append("head_cam_rgb")
            image_key_mapping["head_cam_rgb"] = "observation.images.head_rgb"

        # In compact_actions mode, policy outputs 16-dim (14 arm + 2 gripper)
        upper_body_dof = 16 if self._compact_actions else None

        try:
            self.policy_provider = LeRobotPolicyProvider(
                model_path=self.args.policy_path,
                device=self.args.policy_device,
                use_action_chunking=self.args.use_action_chunking,
                execution_horizon=self.args.execution_horizon,
                image_keys=image_keys if image_keys else ["head_cam_rgb"],
                image_key_mapping=image_key_mapping if image_key_mapping else {"head_cam_rgb": "observation.images.head_rgb"},
                state_key="robot_joint_pos",
                upper_body_dof=upper_body_dof,
                state_dof=None,  # auto-detect from policy config
            )
        except Exception as e:
            print(f"Failed to create policy provider: {e}")
            raise

        if self._compact_actions:
            print(f"  Compact actions mode: 16-dim policy -> 38-dim env actions")
            print(f"    14 arm joints + 2 gripper triggers -> 14 arm + 24 hand joints")

        # Validate policy configuration
        print("\nPolicy information:")
        for key, value in self.policy_provider.policy_info.items():
            print(f"  {key}: {value}")

    def run_episode(self, episode_num: int) -> EpisodeMetrics:
        """Run a single evaluation episode."""
        # Reset environment and policy
        obs_dict, _ = self.env.reset()

        self.policy_provider.reset()

        # Extract observation group for policy (after any randomization)
        obs = self.env.observation_manager.compute_group("policy")

        # Validate observation format on first episode
        if episode_num == 0:
            if not self.policy_provider.validate_observation_format(obs):
                raise ValueError("Observation format validation failed!")

        # Episode tracking
        episode_length = 0
        total_reward = 0.0
        terminated = False
        truncated = False
        inference_times = []

        # Trajectory storage (if enabled)
        trajectory = [] if self.args.save_trajectories else None

        # Video writers (if enabled)
        head_video_writer = None
        scene_video_writer = None
        if self.args.record_video:
            if VIDEO_BACKEND is None:
                print("  Warning: Video recording requested but no backend available. Install imageio or opencv-python.")
            else:
                if self._has_head_cam:
                    head_video_path = Path(self.args.video_dir) / f"episode_{episode_num:03d}_head.mp4"
                    head_video_writer = VideoWriter(head_video_path, fps=self.args.video_fps)
                scene_video_path = Path(self.args.video_dir) / f"episode_{episode_num:03d}_scene.mp4"
                scene_video_writer = VideoWriter(scene_video_path, fps=self.args.video_fps)

        print(f"\nEpisode {episode_num + 1}/{self.args.num_episodes}")

        # Run episode
        while not (terminated or truncated) and episode_length < self.args.max_episode_length:
            # Get action from policy (time it)
            start_time = time.time()
            action = self.policy_provider.get_action(obs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Expand compact 16-dim action to full 38-dim if needed
            if self._compact_actions:
                arm_action = action[:14]
                left_trigger = action[14].item()
                right_trigger = action[15].item()

                # Expand gripper triggers to 24 hand joint positions
                hand_joints = expand_gripper_to_hand_joints(
                    left_trigger, right_trigger, G1_HAND_JOINT_NAMES, device=action.device
                )
                action = torch.cat([arm_action, hand_joints])

                # Copy raw gripper command onto env for the raw_gripper_command observation term
                self.env._raw_gripper_command = torch.tensor(
                    [left_trigger, right_trigger], dtype=torch.float32, device=action.device
                )

            # Expand action to batch dimension for environment
            # (action_dim,) -> (1, action_dim)
            action_batch = action.unsqueeze(0)

            # Step environment
            obs_dict, reward, terminated, truncated, info = self.env.step(action_batch)

            # Update tracking
            episode_length += 1
            total_reward += reward[0].item()

            # Get next observation
            obs = self.env.observation_manager.compute_group("policy")

            # Capture video frames (if recording)
            if head_video_writer is not None and "head_cam_rgb" in obs:
                head_frame = obs["head_cam_rgb"][0].cpu().numpy()
                head_video_writer.add_frame(head_frame)

            if scene_video_writer is not None and "scene_cam" in obs:
                scene_frame = obs["scene_cam"][0].cpu().numpy()
                scene_video_writer.add_frame(scene_frame)

            # Store trajectory data
            if trajectory is not None:
                trajectory.append(
                    {
                        "obs": {k: v.cpu().numpy() for k, v in obs.items()},
                        "action": action.cpu().numpy(),
                        "reward": reward[0].item(),
                    }
                )

            # Print progress
            if episode_length % 50 == 0:
                avg_inference = np.mean(inference_times[-50:])
                print(
                    f"  Step {episode_length:3d} | "
                    f"Reward: {total_reward:7.3f} | "
                    f"Inference: {avg_inference*1000:5.1f}ms"
                )

        # Check success
        terminated_val = terminated[0].item() if isinstance(terminated, torch.Tensor) else terminated
        success = terminated_val and total_reward > 0

        # Save videos if recorded
        if head_video_writer is not None:
            head_video_writer.save()
        if scene_video_writer is not None:
            scene_video_writer.save()

        # Save trajectory if requested
        if self.args.save_trajectories and trajectory:
            self._save_trajectory(episode_num, trajectory)

        # Create metrics
        metrics = EpisodeMetrics(
            episode_num=episode_num,
            episode_length=episode_length,
            total_reward=total_reward,
            success=success,
            inference_time_avg=np.mean(inference_times),
            inference_time_total=np.sum(inference_times),
        )

        return metrics

    def _save_trajectory(self, episode_num: int, trajectory: list):
        """Save trajectory data to disk."""
        import pickle

        traj_path = Path(self.args.trajectory_dir) / f"episode_{episode_num:03d}.pkl"
        with open(traj_path, "wb") as f:
            pickle.dump(trajectory, f)
        print(f"  Saved trajectory to: {traj_path}")

    def evaluate(self):
        """Run full evaluation across all episodes."""
        print("\nStarting evaluation...")

        for episode_num in range(self.args.num_episodes):
            try:
                metrics = self.run_episode(episode_num)
                self.metrics_history.append(metrics)

                # Print episode summary
                print(f"\nEpisode {episode_num + 1} completed:")
                print(f"  Length: {metrics.episode_length}")
                print(f"  Total reward: {metrics.total_reward:.3f}")
                print(f"  Success: {metrics.success}")
                print(f"  Avg inference time: {metrics.inference_time_avg*1000:.1f}ms")

            except KeyboardInterrupt:
                print("\nEvaluation interrupted by user")
                break
            except Exception as e:
                print(f"\nError in episode {episode_num}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Print final summary
        self._print_summary()

    def _print_summary(self):
        """Print summary statistics across all episodes."""
        if not self.metrics_history:
            print("\nNo episodes completed successfully.")
            return

        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        # Aggregate metrics
        episode_lengths = [m.episode_length for m in self.metrics_history]
        total_rewards = [m.total_reward for m in self.metrics_history]
        successes = [m.success for m in self.metrics_history]
        inference_times = [m.inference_time_avg for m in self.metrics_history]

        print(f"Episodes evaluated: {len(self.metrics_history)}")
        print(f"\nSuccess rate: {np.mean(successes)*100:.1f}% ({np.sum(successes)}/{len(successes)})")
        print(f"\nEpisode length:")
        print(f"  Mean: {np.mean(episode_lengths):.1f} steps")
        print(f"  Std:  {np.std(episode_lengths):.1f} steps")
        print(f"  Min:  {np.min(episode_lengths)} steps")
        print(f"  Max:  {np.max(episode_lengths)} steps")
        print(f"\nTotal reward:")
        print(f"  Mean: {np.mean(total_rewards):.3f}")
        print(f"  Std:  {np.std(total_rewards):.3f}")
        print(f"  Min:  {np.min(total_rewards):.3f}")
        print(f"  Max:  {np.max(total_rewards):.3f}")
        print(f"\nInference time:")
        print(f"  Mean: {np.mean(inference_times)*1000:.1f}ms")
        print(f"  Std:  {np.std(inference_times)*1000:.1f}ms")
        print(f"  Min:  {np.min(inference_times)*1000:.1f}ms")
        print(f"  Max:  {np.max(inference_times)*1000:.1f}ms")

        if self.args.use_action_chunking:
            avg_length = np.mean(episode_lengths)
            num_inferences = avg_length / self.args.execution_horizon
            print(f"\nAction chunking efficiency:")
            print(f"  Inferences per episode: ~{num_inferences:.0f}")
            print(f"  Reduction factor: {self.args.execution_horizon}x")

        print("=" * 70)

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        if hasattr(self, "env"):
            self.env.close()


def main():
    """Main entry point."""
    evaluator = None
    try:
        evaluator = PolicyEvaluator(args_cli)
        evaluator.evaluate()
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if evaluator is not None:
            evaluator.cleanup()

    return 0


if __name__ == "__main__":
    # run the main function
    exit_code = main()
    # close sim app
    simulation_app.close()
    exit(exit_code)
