# Mink IK State Synchronization Integration Guide

This guide explains how to integrate the Mink IK state synchronization feature into your Isaac Lab teleoperation workflows.

## Overview

The Mink IK retargeter runs in a separate thread with its own MuJoCo simulation state. Over time, this internal state can drift from the actual Isaac Lab simulation, causing inconsistent behavior and jerky motions. The state synchronization feature solves this by:

1. Providing measured joint positions from Isaac Lab to the Mink IK solver
2. Automatically syncing the internal Mink state when hands are not actively controlling
3. Resetting the Mink state properly on environment resets

## Quick Start

### Minimal Integration

Here's the minimal code needed to enable state synchronization:

```python
# Get measured joint positions from robot
upper_body_joint_ids = robot.find_joints(upper_body_joint_names, preserve_order=True)[0]
measured_joint_pos = robot.data.joint_pos[0, upper_body_joint_ids]

# Provide to device before getting actions
device.set_measured_joint_positions(measured_joint_pos)

# Get actions as normal
actions = device.advance()
```

### With Reset Handling

For proper reset behavior, also sync on environment reset:

```python
# On environment reset
env.reset()

# Get current joint positions
measured_joint_pos = robot.data.joint_pos[0, upper_body_joint_ids]

# Update device
device.set_measured_joint_positions(measured_joint_pos)

# Reset retargeters
for retargeter in device._retargeters:
    if hasattr(retargeter, 'reset'):
        retargeter.reset(measured_joint_pos.cpu().numpy())
```

## Complete Integration Examples

### Example 1: Standalone Script

```python
#!/usr/bin/env python3
"""Example teleoperation script with state sync."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.devices.teleop_device_factory import create_teleop_device

def main():
    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Create teleop device
    device = create_teleop_device("xr_controller", env_cfg.teleop_devices.devices)

    # Get robot and joint indices
    robot = env.scene["robot"]
    upper_body_joint_names = [
        "AAHead_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
    ]
    upper_body_joint_ids = robot.find_joints(upper_body_joint_names, preserve_order=True)[0]

    while simulation_app.is_running():
        with torch.inference_mode():
            # On reset
            if should_reset:
                env.reset()
                measured_joint_pos = robot.data.joint_pos[0, upper_body_joint_ids]
                device.set_measured_joint_positions(measured_joint_pos)

                for retargeter in device._retargeters:
                    if hasattr(retargeter, 'reset'):
                        retargeter.reset(measured_joint_pos.cpu().numpy())

            # Get measured joint positions
            measured_joint_pos = robot.data.joint_pos[0, upper_body_joint_ids]

            # Update device state
            device.set_measured_joint_positions(measured_joint_pos)

            # Get actions
            actions = device.advance()

            # Step environment
            obs, rew, terminated, truncated, info = env.step(actions)

    env.close()

if __name__ == "__main__":
    main()
```

### Example 2: Modifying record_demos.py

To integrate into the `scripts/tools/record_demos.py` script:

```python
# Add after creating the teleop device (around line 267):
# Get robot and joint indices for state sync
robot = env.scene["robot"]
upper_body_joint_names = [
    "AAHead_yaw", "Head_pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
]
upper_body_joint_ids = robot.find_joints(upper_body_joint_names, preserve_order=True)[0]

# Modify the reset callback (around line 403):
def reset_recording_instance():
    global running_recording_instance, should_reset_recording_instance, success_step_count
    running_recording_instance = False
    should_reset_recording_instance = True
    success_step_count = 0

    # Add state sync on reset
    env.sim.reset()
    env.reset()
    measured_joint_pos = robot.data.joint_pos[0, upper_body_joint_ids]
    teleop_interface.set_measured_joint_positions(measured_joint_pos)

    for retargeter in teleop_interface._retargeters:
        if hasattr(retargeter, 'reset'):
            retargeter.reset(measured_joint_pos.cpu().numpy())

    teleop_interface.reset()

# Modify the main loop (before line 445):
# Get measured joint positions
measured_joint_pos = robot.data.joint_pos[0, upper_body_joint_ids]

# Update device state
teleop_interface.set_measured_joint_positions(measured_joint_pos)

# Get keyboard command
action = teleop_interface.advance()
```

## Joint Names Reference

The upper body joint names for T1 robot are:

```python
upper_body_joint_names = [
    # Head (2 joints)
    "AAHead_yaw", "Head_pitch",

    # Left arm (7 joints)
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",

    # Right arm (7 joints)
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
]
```

Total: **16 joints**

## How It Works

### 1. State Sync During Control

When you call `device.set_measured_joint_positions(measured_joint_pos)`:
- The device stores the measured positions
- On the next call to `device.advance()`, it includes these in the data passed to retargeters
- The Mink IK retargeter stores these measured positions

### 2. Automatic Synchronization

When grip is pressed on either hand:
- **On grip press transition** (released → pressed): The solver syncs its internal MuJoCo state with the measured positions
- **While gripping**: Delta tracking from the anchored state is used for smooth control
- **When released**: No syncing occurs to avoid drift
- This anchors the IK state to the simulation state at the moment of grip press

### 3. Reset Behavior

When you call `retargeter.reset(joint_positions)`:
- Updates internal MuJoCo state to match provided positions
- Resets mocap targets to current hand positions
- Clears delta tracking references
- Ensures clean state after environment reset

## Benefits

- **Prevents drift**: State sync on grip press anchors IK to simulation state at control initiation
- **Smooth control**: Delta tracking from anchored state provides responsive control
- **Clean resets**: Proper state initialization on environment reset
- **Non-intrusive**: Minimal code changes required
- **Thread-safe**: All updates protected by existing locks

## Troubleshooting

### State sync not working
- Verify joint names match your robot configuration
- Check that `device.set_measured_joint_positions()` is called before `device.advance()`
- Ensure measured positions are a 16-element tensor/array

### Jerky motion persists
- Verify state sync is being called in the main loop (not just on reset)
- Check that joint IDs are found correctly (no errors in joint lookup)
- Ensure device has Mink IK retargeters configured

### Reset issues
- Call `retargeter.reset()` with measured positions after `env.reset()`
- Verify retargeter.reset() is called for all retargeters in the device

## API Reference

### XRControllerDevice

```python
def set_measured_joint_positions(self, joint_positions: torch.Tensor | np.ndarray):
    """Set measured joint positions from simulation.

    Args:
        joint_positions: 16-element tensor/array of upper body joint positions
    """
```

### XRT1MinkIKRetargeter

```python
def reset(self, joint_positions: np.ndarray | None = None):
    """Reset IK solver state.

    Args:
        joint_positions: Optional 16-element array of joint positions.
                        If None, resets to home position.
    """

def set_qpos_upper(self, joint_positions: np.ndarray):
    """Update internal MuJoCo state.

    Args:
        joint_positions: 16-element array of joint positions
    """
```

## See Also

- `test_t1_mink_state_sync.py` - Complete working example
- `teleop_device_sync.py` - Helper functions for environment integration
- XRoboToolkit documentation - Original pattern inspiration
