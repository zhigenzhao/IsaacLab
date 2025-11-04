#!/bin/bash
# Example usage scripts for evaluating LeRobot policies in Isaac Lab
# Make this file executable: chmod +x scripts/tools/example_eval_lerobot.sh

# === BASIC EXAMPLES ===

# Example 1: Basic evaluation (10 episodes)
echo "Example 1: Basic evaluation"
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-T1-Stack-Cube-v0 \
    --policy_path kelvinzhaozg/t1_stack_cube_policy \
    --num_episodes 10 \
    --policy_device cuda

# Example 2: With action chunking (recommended for faster inference)
echo "Example 2: With action chunking"
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-T1-Stack-Cube-v0 \
    --policy_path kelvinzhaozg/t1_stack_cube_policy \
    --num_episodes 10 \
    --use_action_chunking \
    --execution_horizon 8 \
    --policy_device cuda

# Example 3: With video recording (cameras automatically enabled)
echo "Example 3: With video recording"
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-T1-Stack-Cube-v0 \
    --policy_path kelvinzhaozg/t1_stack_cube_policy \
    --num_episodes 5 \
    --use_action_chunking \
    --record_video \
    --video_dir ./evaluation_videos \
    --video_fps 30 \
    --policy_device cuda

# Example 3b: Headless with video recording (cameras still work!)
echo "Example 3b: Headless with video recording"
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-T1-Stack-Cube-v0 \
    --policy_path kelvinzhaozg/t1_stack_cube_policy \
    --num_episodes 5 \
    --use_action_chunking \
    --record_video \
    --video_dir ./evaluation_videos_headless \
    --headless \
    --policy_device cuda

# Example 4: Headless evaluation with trajectory saving
echo "Example 4: Headless with trajectories"
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-T1-Stack-Cube-v0 \
    --policy_path kelvinzhaozg/t1_stack_cube_policy \
    --num_episodes 20 \
    --use_action_chunking \
    --save_trajectories \
    --trajectory_dir ./trajectories \
    --headless \
    --policy_device cuda

# === ADVANCED EXAMPLES ===

# Example 5: Local policy evaluation
echo "Example 5: Local policy"
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-T1-Stack-Cube-v0 \
    --policy_path /path/to/local/policy \
    --num_episodes 10 \
    --use_action_chunking \
    --policy_device cuda

# Example 6: CPU evaluation (no CUDA)
echo "Example 6: CPU evaluation"
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-T1-Stack-Cube-v0 \
    --policy_path kelvinzhaozg/t1_stack_cube_policy \
    --num_episodes 5 \
    --use_action_chunking \
    --device cpu

# Example 7: Quick test (1 episode, short)
echo "Example 7: Quick test"
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-T1-Stack-Cube-v0 \
    --policy_path kelvinzhaozg/t1_stack_cube_policy \
    --num_episodes 1 \
    --max_episode_length 100 \
    --use_action_chunking \
    --policy_device cuda

# Example 8: Long evaluation with full logging
echo "Example 8: Long evaluation"
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-T1-Stack-Cube-v0 \
    --policy_path kelvinzhaozg/t1_stack_cube_policy \
    --num_episodes 50 \
    --max_episode_length 1000 \
    --use_action_chunking \
    --execution_horizon 8 \
    --save_trajectories \
    --trajectory_dir ./long_eval_trajectories \
    --seed 42 \
    --headless \
    --policy_device cuda
