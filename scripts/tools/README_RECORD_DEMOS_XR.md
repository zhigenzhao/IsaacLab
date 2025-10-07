# record_demos_xr.py - XR Teleoperation with State Synchronization

This script is an enhanced version of `record_demos.py` that includes Mink IK state synchronization for XR controller teleoperation. It's specifically designed for recording demonstrations with humanoid robots like T1 using XR controllers.

## Key Features

- **Automatic State Synchronization**: Continuously syncs the Mink IK solver internal state with the actual Isaac Lab simulation
- **Reset State Sync**: Properly resets IK solver state on environment resets
- **Drift Prevention**: Prevents inconsistencies between IK solver and simulation that cause jerky motions
- **Backward Compatible**: Works with all existing teleop devices; state sync only activates when applicable

## Usage

### Basic Usage

Record demonstrations with XR controller for T1 manipulation environment:

```bash
python scripts/tools/record_demos_xr.py \
    --task Isaac-T1-Teleop-Manipulation \
    --teleop_device xr_controller \
    --num_demos 10
```

### With Custom Dataset Path

```bash
python scripts/tools/record_demos_xr.py \
    --task Isaac-T1-Teleop-Manipulation \
    --teleop_device xr_controller \
    --dataset_file ./my_datasets/t1_manipulation_demos.hdf5 \
    --num_demos 20
```

### All Options

```bash
python scripts/tools/record_demos_xr.py \
    --task TASK_NAME \
    --teleop_device DEVICE_NAME \
    --dataset_file PATH_TO_HDF5 \
    --step_hz HZ \
    --num_demos COUNT \
    --num_success_steps STEPS
```

## Arguments

### Required
- `--task`: Name of the task (e.g., `Isaac-T1-Teleop-Manipulation`)

### Optional
- `--teleop_device`: Device for teleoperation (default: `keyboard`)
  - For XR: `xr_controller`
  - For keyboard: `keyboard`
  - For spacemouse: `spacemouse`
- `--dataset_file`: Output HDF5 file path (default: `./datasets/dataset.hdf5`)
- `--step_hz`: Environment stepping rate in Hz (default: `30`)
- `--num_demos`: Number of demonstrations to record (default: `0` for infinite)
- `--num_success_steps`: Consecutive success steps required (default: `10`)

## How State Synchronization Works

### 1. Initialization
On startup, the script:
- Identifies the robot in the scene
- Finds upper body joint indices (16 joints for T1)
- Enables state sync logging

### 2. During Teleoperation
Before each `teleop_interface.advance()` call:
- Reads current joint positions from simulation
- Passes them to the device via `set_measured_joint_positions()`
- The Mink IK retargeter syncs its internal state when grip is first pressed (on the transition from released to pressed)
- This anchors the IK state to the current simulation state at the moment of control initiation

### 3. On Reset
When environment resets (R key or automatic):
- Resets environment
- Reads current joint positions
- Resets all retargeters with measured positions
- Ensures clean IK solver state

## Supported Robots

The state synchronization currently works with robots that have these upper body joints:

**T1 Humanoid (16 joints)**:
- Head: `AAHead_yaw`, `Head_pitch`
- Left Arm: `Left_Shoulder_Pitch`, `Left_Shoulder_Roll`, `Left_Elbow_Pitch`, `Left_Elbow_Yaw`, `Left_Wrist_Pitch`, `Left_Wrist_Yaw`, `Left_Hand_Roll`
- Right Arm: `Right_Shoulder_Pitch`, `Right_Shoulder_Roll`, `Right_Elbow_Pitch`, `Right_Elbow_Yaw`, `Right_Wrist_Pitch`, `Right_Wrist_Yaw`, `Right_Hand_Roll`

For other robots, modify the `upper_body_joint_names` list in the script (line 442).

## Differences from record_demos.py

1. **State Sync Setup** (lines 436-471): Initializes state sync infrastructure
2. **Pre-Step Sync** (lines 480-484): Updates device state before getting actions
3. **Reset Sync** (lines 391-400): Syncs state on environment reset
4. **Enhanced Logging**: Logs when state sync is enabled/disabled

## When to Use This Script vs. record_demos.py

### Use record_demos_xr.py when:
- Using XR controller with Mink IK retargeter
- Working with T1 or other humanoid robots
- Experiencing drift or jerky motions during teleoperation
- Recording long demonstration sequences

### Use original record_demos.py when:
- Using keyboard or spacemouse
- Working with robots that don't need IK state sync
- You want to avoid any custom modifications

## Troubleshooting

### State sync not enabled
Check the console output at startup:
```
[Info] Mink IK state synchronization enabled for XR teleoperation
```

If you see:
```
[Warning] Could not find upper body joints for state sync
```

Then the robot doesn't have the expected joint names. Modify `upper_body_joint_names` in the script.

### Still experiencing drift
1. Verify state sync is enabled (check logs)
2. Ensure XR controller device is being used
3. Check that Mink IK retargeter is configured in environment
4. Try increasing `--step_hz` for faster sync updates

### Performance issues
If state sync causes performance degradation:
1. The sync overhead is minimal (~0.1ms per step)
2. Check if other factors are affecting performance
3. State sync only activates when applicable (has no effect on non-IK devices)

## Examples

### Record 10 T1 Manipulation Demos
```bash
python scripts/tools/record_demos_xr.py \
    --task Isaac-T1-Teleop-Manipulation \
    --teleop_device xr_controller \
    --num_demos 10 \
    --dataset_file ./datasets/t1_manip_demos.hdf5
```

### Record Until Manual Stop
```bash
python scripts/tools/record_demos_xr.py \
    --task Isaac-T1-Teleop-Manipulation \
    --teleop_device xr_controller
```
(Press Ctrl+C to stop)

### Record with Higher Update Rate
```bash
python scripts/tools/record_demos_xr.py \
    --task Isaac-T1-Teleop-Manipulation \
    --teleop_device xr_controller \
    --step_hz 50
```

## See Also

- `record_demos.py` - Original script without state sync
- `test_t1_mink_state_sync.py` - Minimal state sync example
- `docs/MINK_STATE_SYNC_INTEGRATION.md` - Detailed integration guide
