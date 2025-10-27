# LeRobot Policy Integration for Isaac Lab

This directory contains tools for evaluating pretrained [LeRobot](https://github.com/huggingface/lerobot) policies in Isaac Lab environments.

## Components

### 1. `lerobot_action_chunk_broker.py`
Manages action chunking to reduce policy inference frequency. Caches action sequences and returns them sequentially until exhausted, then triggers new inference.

**Key features:**
- Configurable execution horizon (default: 8 steps between inferences)
- Automatic cache management
- Works with any LeRobot policy type

### 2. `lerobot_policy_provider.py`
Main interface between LeRobot policies and Isaac Lab environments. Handles:
- Loading pretrained policies from HuggingFace Hub or local paths
- Converting Isaac Lab observations → LeRobot format
- Converting LeRobot actions → Isaac Lab joint positions
- Optional action chunking integration

**Key features:**
- Policy-agnostic design (works with diffusion, ACT, VQ-BeT, flow matching, etc.)
- Automatic observation format conversion
- Device management (CUDA, CPU, MPS)
- Rich debugging information

### 3. `eval_lerobot_policy.py`
Standalone evaluation script for running policy rollouts in Isaac Lab.

**Key features:**
- Episode-based evaluation with detailed metrics
- Optional video recording
- Optional trajectory saving
- Success rate tracking
- Inference timing analysis

## Installation

Ensure you have Isaac Lab and LeRobot installed:

```bash
# Isaac Lab should already be installed
# Install LeRobot
pip install lerobot

# For video recording (choose one):
pip install imageio[ffmpeg]  # Recommended
# OR
pip install opencv-python
```

## Usage

### Basic Evaluation

Evaluate a pretrained policy from HuggingFace Hub:

```bash
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-Stack-Cube-T1-v0 \
    --policy_path kelvinzhaozg/t1_stack_cube_policy \
    --num_episodes 10 \
    --policy_device cuda
```

### With Action Chunking

Enable action chunking to reduce inference frequency (8x reduction):

```bash
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-Stack-Cube-T1-v0 \
    --policy_path kelvinzhaozg/t1_stack_cube_policy \
    --num_episodes 10 \
    --use_action_chunking \
    --execution_horizon 8 \
    --policy_device cuda
```

### With Video Recording

Record videos of rollouts (cameras are automatically enabled):

```bash
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-Stack-Cube-T1-v0 \
    --policy_path kelvinzhaozg/t1_stack_cube_policy \
    --num_episodes 5 \
    --record_video \
    --video_dir ./evaluation_videos \
    --video_fps 30 \
    --policy_device cuda
```

### Headless Mode with Video Recording

**Important**: You can record videos even in headless mode! Cameras will render without the GUI:

```bash
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-Stack-Cube-T1-v0 \
    --policy_path kelvinzhaozg/t1_stack_cube_policy \
    --num_episodes 10 \
    --record_video \
    --video_dir ./evaluation_videos \
    --headless \
    --policy_device cuda
```

This is useful for:
- Running evaluations on remote servers without display
- Faster evaluation (no GUI overhead)
- Batch evaluation jobs

### With Trajectory Saving

Save trajectory data for analysis:

```bash
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-Stack-Cube-T1-v0 \
    --policy_path kelvinzhaozg/t1_stack_cube_policy \
    --num_episodes 10 \
    --save_trajectories \
    --trajectory_dir ./trajectories \
    --policy_device cuda
```

### Local Policy

Evaluate a policy from a local directory:

```bash
python scripts/tools/eval_lerobot_policy.py \
    --task Isaac-Stack-Cube-T1-v0 \
    --policy_path /path/to/local/policy \
    --num_episodes 10 \
    --policy_device cuda
```

## Command-Line Arguments

### Required
- `--task`: Isaac Lab task name (e.g., `Isaac-Stack-Cube-T1-v0`)
- `--policy_path`: Path to pretrained policy (HuggingFace Hub ID or local path)

### Evaluation Settings
- `--num_episodes`: Number of episodes to evaluate (default: 10)
- `--max_episode_length`: Maximum steps per episode (default: 500)
- `--seed`: Random seed for reproducibility (default: 42)

### Policy Settings
- `--policy_device`: Device for policy inference - `cuda`, `cpu`, or `mps` (default: cuda)
- `--use_action_chunking`: Enable action chunking (flag)
- `--execution_horizon`: Steps before re-inference when chunking (default: 8)

**Note**: Isaac Lab uses `--device` for the simulation device. Policy inference uses a separate `--policy_device` flag.

### Recording Settings
- `--record_video`: Record video of rollouts (flag, automatically enables cameras)
- `--video_dir`: Directory to save videos (default: ./videos)
- `--video_fps`: FPS for recorded videos (default: 30)
- `--save_trajectories`: Save trajectory data (flag)
- `--trajectory_dir`: Directory to save trajectories (default: ./trajectories)

**Note**: When `--record_video` is enabled, cameras are automatically enabled even in headless mode. Videos are saved as MP4 files using imageio or OpenCV backend.

### Isaac Sim Settings
All standard Isaac Lab AppLauncher arguments are supported:
- `--headless`: Run without GUI
- `--enable_cameras`: Enable camera rendering
- etc.

## Expected Dataset Format

The policy should be trained on data following this format:

```python
{
    "observation.images.head_rgb": (height, width, 3),  # RGB image, uint8
    "observation.state": (18,),                          # Upper body joint positions, float32
    "action": (18,),                                     # Upper body joint positions, float32
}
```

This matches the output from `convert_isaaclab_data_to_lerobot.py` in the diffusion_planner project.

## Observation/Action Space Mapping

### Isaac Lab → LeRobot

**Observations:**
```python
# Isaac Lab format (from observation_manager)
{
    "head_rgb_cam": (num_envs, H, W, 3),  # uint8
    "joint_pos": (num_envs, 21),           # float32, all joints
}

# LeRobot format (after preprocessing)
{
    "observation.images.head_rgb": (1, 3, H, W),  # float32 [0, 1], channels-first
    "observation.state": (1, 18),                  # float32, upper body only
}
```

**Actions:**
```python
# LeRobot format (policy output)
(18,)  # Upper body joint positions, float32

# Isaac Lab format (for env.step)
(num_envs, 18)  # Batched actions, float32
```

## Testing

Run the test suite to validate the integration:

```bash
python scripts/tools/test_lerobot_integration.py
```

This tests:
1. Import validation
2. Action chunk broker functionality
3. Observation preprocessing

## Example Output

```
======================================================================
LeRobot Policy Evaluation in Isaac Lab
======================================================================
Task: Isaac-Stack-Cube-T1-v0
Policy: kelvinzhaozg/t1_stack_cube_policy
Episodes: 10
Max episode length: 500
Action chunking: True
Execution horizon: 8
======================================================================

Episode 1/10
  Step  50 | Reward:   0.123 | Inference:  12.3ms
  Step 100 | Reward:   0.456 | Inference:  11.8ms
  ...

Episode 1 completed:
  Length: 234
  Total reward: 1.234
  Success: True
  Avg inference time: 12.1ms

...

======================================================================
EVALUATION SUMMARY
======================================================================
Episodes evaluated: 10

Success rate: 80.0% (8/10)

Episode length:
  Mean: 245.3 steps
  Std:  23.4 steps
  Min:  198 steps
  Max:  289 steps

Total reward:
  Mean: 1.567
  Std:  0.234
  Min:  1.123
  Max:  1.890

Inference time:
  Mean: 12.3ms
  Std:  1.2ms
  Min:  10.5ms
  Max:  15.8ms

Action chunking efficiency:
  Inferences per episode: ~31
  Reduction factor: 8x
======================================================================
```

## Supported Policy Types

The integration is policy-agnostic and supports all LeRobot policy types:
- ✅ Diffusion Policy
- ✅ Diffusion Transformer
- ✅ Flow Matching
- ✅ ACT (Action Chunking Transformer)
- ✅ VQ-BeT
- ✅ VQFlow
- ✅ PI0
- ✅ SmolVLA
- ✅ Groot

The policy type is automatically detected from the configuration file.

## Architecture

The system follows a hierarchical design inspired by the thirdarm_project:

```
LeRobot Policy (pretrained)
    ↓
ActionChunkBroker (optional)
    ↓
LeRobotPolicyProvider
    ↓
Isaac Lab Environment
```

**Key differences from thirdarm:**
- No frequency synchronization (Isaac Lab runs at fixed rate)
- No real-time threading (simpler, single-threaded)
- Direct joint position control (no IK solving)
- Policy-agnostic design (works with any LeRobot policy)

## Troubleshooting

### Import Errors
Ensure LeRobot is installed:
```bash
pip install lerobot
```

### Policy Loading Errors
- Check that the policy path is correct (HuggingFace Hub ID or local directory)
- Verify the policy contains `config.json` and `model.safetensors`
- Check network connection for Hub downloads

### Observation Format Mismatch
The policy provider will validate observation format on the first episode and print warnings if issues are detected. Common issues:
- Missing camera or state observations
- Wrong image dimensions (should be 240x424 for T1 tasks)
- Insufficient joint dimensions (needs at least 18 for upper body)

### Action Dimension Mismatch
If you see action dimension errors, verify:
- Policy was trained with 18D actions (upper body joints)
- Dataset was created with correct action dimensions
- Check `upper_body_dof` parameter in PolicyProvider

### Video Recording Issues

**No video backend available:**
```bash
pip install imageio[ffmpeg]  # Recommended
# OR
pip install opencv-python
```

**Videos not recording in headless mode:**
- This should work automatically! The `--record_video` flag automatically enables cameras
- Check that the video directory is writable
- Verify the episode runs to completion (videos are saved at the end)

**Videos are empty or corrupted:**
- Check that camera observations are being captured (enable verbose logging)
- Verify camera is configured in the environment
- Try a different video backend (imageio vs opencv)

## Advanced Usage

### Custom Observation Keys

If your environment uses different observation keys:

```python
from scripts.tools.lerobot_policy_provider import LeRobotPolicyProvider

provider = LeRobotPolicyProvider(
    model_path="path/to/policy",
    device="cuda",
    image_key="my_camera",        # Custom camera name
    state_key="my_joint_state",   # Custom state name
    upper_body_dof=16,            # Custom action dimension
)
```

### Programmatic Usage

Use the policy provider directly in your code:

```python
import torch
from scripts.tools.lerobot_policy_provider import LeRobotPolicyProvider

# Initialize provider
provider = LeRobotPolicyProvider(
    model_path="kelvinzhaozg/t1_stack_cube_policy",
    device="cuda",
    use_action_chunking=True,
    execution_horizon=8,
)

# Reset for new episode
provider.reset()

# Get observation from environment
obs = env.observation_manager.compute_group("policy")

# Get action
action = provider.get_action(obs)  # Returns (18,) numpy array

# Step environment
env.step(action[None, :])  # Add batch dimension
```

## Performance Tips

1. **Use action chunking** for policies that support it (most do). This reduces inference frequency by 8x with minimal performance loss.

2. **Use CUDA** if available. Policy inference is much faster on GPU.

3. **Batch size 1**: The current implementation only supports single-environment evaluation (`num_envs=1`). This is intentional for evaluation clarity.

4. **Profile inference**: Use `--save_trajectories` to analyze timing and identify bottlenecks.

## Future Enhancements

Potential improvements for future iterations:

- [ ] Real-time action chunking with threading (as in thirdarm)
- [ ] Frequency synchronization for smooth interpolation
- [ ] Multi-environment support for parallel evaluation
- [ ] End-effector control mode (with IK solving)
- [ ] Integration as Isaac Lab Manager components
- [ ] Automatic hyperparameter tuning
- [ ] Live visualization dashboard

## References

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Thirdarm Project](https://github.com/zhigenzhao/thirdarm_project) (inspiration for architecture)

## Contributing

To extend this integration:

1. **Add new observation types**: Modify `preprocess_observation()` in `lerobot_policy_provider.py`
2. **Add new action types**: Modify `get_action()` to handle different control modes
3. **Add new metrics**: Extend `EpisodeMetrics` in `eval_lerobot_policy.py`
4. **Add new policy types**: Should work automatically through LeRobot's `PreTrainedPolicy` interface

## License

This code follows the same BSD-3-Clause license as Isaac Lab.
