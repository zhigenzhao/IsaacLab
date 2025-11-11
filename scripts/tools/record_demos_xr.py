# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Script to record demonstrations with Isaac Lab environments using XR controller teleoperation with state synchronization.

This script includes Mink IK state synchronization for improved tracking consistency with XR controller
teleoperation. It prevents drift between the Mink IK solver internal state and the actual Isaac Lab
simulation state.

This script allows users to record demonstrations operated by XR controller teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, dataset directory,
and environment stepping rate through command-line arguments.

Only XR controller teleoperation is supported by this script.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: xr_controller, only xr_controller is supported)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30, note: rate limiting is handled by OpenXR)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
import argparse
import contextlib
from datetime import datetime

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="xr_controller", help="Device for interacting with environment (only xr_controller is supported).")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=50, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Validate required arguments
if args_cli.task is None:
    parser.error("--task is required")

app_launcher_args = vars(args_cli)

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


# Third-party imports
import gymnasium as gym
import os
import time
import torch

# Omniverse logger
import omni.log
import omni.ui as ui

from isaaclab.devices.teleop_device_factory import create_teleop_device

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

from collections.abc import Callable

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories() -> tuple[str, str]:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist and extracts the file name
    from the dataset file path. Adds a timestamp to the filename to prevent overwriting.

    Returns:
        tuple[str, str]: A tuple containing:
            - output_dir: The directory path where the dataset will be saved
            - output_file_name: The filename (without extension) with timestamp for the dataset
    """
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # Add timestamp to filename to prevent overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_name = f"{output_file_name}_{timestamp}"

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Dataset will be saved as: {output_file_name}.hdf5")

    return output_dir, output_file_name


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None]:
    """Create and configure the environment configuration.

    Parses the environment configuration and makes necessary adjustments for demo recording.
    Extracts the success termination function and configures the recorder manager.

    Args:
        output_dir: Directory where recorded demonstrations will be saved
        output_file_name: Name of the file to store the demonstrations

    Returns:
        tuple[isaaclab_tasks.utils.parse_cfg.EnvCfg, Optional[object]]: A tuple containing:
            - env_cfg: The configured environment configuration
            - success_term: The success termination object or None if not available

    Raises:
        Exception: If parsing the environment configuration fails
    """
    # parse configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as e:
        omni.log.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # Enable DLSS antialiasing for XR controller
    # Cameras are enabled for demonstration recording to record visual observations
    env_cfg.sim.render.antialiasing_mode = "DLSS"

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def create_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    """Create the environment from the configuration.

    Args:
        env_cfg: The environment configuration object that defines the environment properties.
            This should be an instance of EnvCfg created by parse_env_cfg().

    Returns:
        gym.Env: A Gymnasium environment instance for the specified task.

    Raises:
        Exception: If environment creation fails for any reason.
    """
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        return env
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        exit(1)


def setup_teleop_device(callbacks: dict[str, Callable]) -> object:
    """Set up the XR controller teleoperation device.

    Creates an XR controller device from the environment configuration.

    Args:
        callbacks: Dictionary mapping callback keys to functions that will be
                   attached to the teleop device

    Returns:
        object: The configured XR controller teleoperation device interface

    Raises:
        SystemExit: If teleop device creation fails or xr_controller is not configured
    """
    try:
        if not hasattr(env_cfg, "teleop_devices") or args_cli.teleop_device not in env_cfg.teleop_devices.devices:
            omni.log.error(f"No '{args_cli.teleop_device}' found in environment config.")
            omni.log.error("Ensure the environment has XR controller support configured.")
            exit(1)

        teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)

        if teleop_interface is None:
            omni.log.error(f"Failed to create {args_cli.teleop_device} interface")
            exit(1)

        return teleop_interface

    except Exception as e:
        omni.log.error(f"Failed to create teleop device: {e}")
        exit(1)


def setup_ui(label_text: str, env: gym.Env) -> InstructionDisplay:
    """Set up the user interface elements.

    Creates instruction display and UI window with labels for showing information
    to the user during demonstration recording.

    Args:
        label_text: Text to display showing current recording status
        env: The environment instance for which UI is being created

    Returns:
        InstructionDisplay: The configured instruction display object
    """
    instruction_display = InstructionDisplay(xr=False)
    window = EmptyWindow(env, "Instruction")
    with window.ui_window_elements["main_vstack"]:
        demo_label = ui.Label(label_text)
        subtask_label = ui.Label("")
        instruction_display.set_labels(subtask_label, demo_label)

    return instruction_display


def process_success_condition(env: gym.Env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
    """Process the success condition for the current step.

    Checks if the environment has met the success condition for the required
    number of consecutive steps. Marks the episode as successful if criteria are met.

    Args:
        env: The environment instance to check
        success_term: The success termination object or None if not available
        success_step_count: Current count of consecutive successful steps

    Returns:
        tuple[int, bool]: A tuple containing:
            - updated success_step_count: The updated count of consecutive successful steps
            - success_reset_needed: Boolean indicating if reset is needed due to success
    """
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


def handle_reset(
    env: gym.Env,
    success_step_count: int,
    instruction_display: InstructionDisplay,
    label_text: str,
    teleop_interface: object | None = None,
    robot: object | None = None,
    upper_body_joint_ids: torch.Tensor | None = None,
) -> int:
    """Handle resetting the environment.

    Resets the environment, recorder manager, and related state variables.
    Updates the instruction display with current status.
    Syncs teleop device state if applicable.

    Args:
        env: The environment instance to reset
        success_step_count: Current count of consecutive successful steps
        instruction_display: The display object to update
        label_text: Text to display showing current recording status
        teleop_interface: Optional teleop interface for state sync
        robot: Optional robot asset for reading joint positions
        upper_body_joint_ids: Optional joint IDs for state sync

    Returns:
        int: Reset success step count (0)
    """
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    success_step_count = 0
    instruction_display.show_demo(label_text)

    # Sync teleop device state after reset (for Mink IK state sync)
    if teleop_interface is not None and robot is not None and upper_body_joint_ids is not None:
        measured_joint_pos = robot.data.joint_pos[0, upper_body_joint_ids]
        if hasattr(teleop_interface, 'set_measured_joint_positions'):
            teleop_interface.set_measured_joint_positions(measured_joint_pos)
        if hasattr(teleop_interface, '_retargeters'):
            for retargeter in teleop_interface._retargeters:
                if hasattr(retargeter, 'reset'):
                    # Pass joint positions as optional kwarg - retargeters use it if needed
                    retargeter.reset(joint_positions=measured_joint_pos.cpu().numpy())
                    print(f"  Reset {type(retargeter).__name__} with state sync")

    return success_step_count


def run_simulation_loop(
    env: gym.Env,
    teleop_interface: object | None,
    success_term: object | None,
    rate_limiter: RateLimiter | None,
) -> int:
    """Run the main simulation loop for collecting demonstrations.

    Sets up callback functions for teleop device, initializes the UI,
    and runs the main loop that processes user inputs and environment steps.
    Records demonstrations when success conditions are met.

    Args:
        env: The environment instance
        teleop_interface: Optional teleop interface (will be created if None)
        success_term: The success termination object or None if not available
        rate_limiter: Optional rate limiter to control simulation speed

    Returns:
        int: Number of successful demonstrations recorded
    """
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False
    # XR controller starts in paused state
    running_recording_instance = False

    # Callback closures for the teleop device
    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("🔄 Recording instance reset requested")

    def start_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = True
        # Capture initial_state when recording actually begins
        env.recorder_manager.record_initial_state([0])
        print("▶️  Recording started")

    def pause_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("⏸️  Recording paused")

    def save_recording_instance():
        nonlocal should_reset_recording_instance, running_recording_instance
        if running_recording_instance or env.recorder_manager.get_episode(0).length() > 0:
            print("💾 Saving current demonstration...")
            # Stop recording
            running_recording_instance = False
            # Mark episode as successful and export
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("✅ Episode saved successfully!")
            # Trigger reset
            should_reset_recording_instance = True
        else:
            print("ℹ️  No data to save - episode is empty")

    def discard_recording_instance():
        nonlocal should_reset_recording_instance, running_recording_instance
        if running_recording_instance:
            print("🗑️  Discarding current demonstration (not saved)")
            running_recording_instance = False
            should_reset_recording_instance = True
        else:
            print("ℹ️  Nothing to discard - not currently recording")

    # Set up teleoperation callbacks
    # Button mapping: A=START, B=SAVE, X=RESET, Y=PAUSE, Right-stick-click=DISCARD
    teleoperation_callbacks = {
        "R": reset_recording_instance,          # Keyboard R
        "START": start_recording_instance,      # A button (right_primary)
        "SAVE": save_recording_instance,        # B button (right_secondary)
        "RESET": reset_recording_instance,      # X button (left_primary)
        "PAUSE": pause_recording_instance,      # Y button (left_secondary)
        "DISCARD": discard_recording_instance,  # Right joystick click
    }

    teleop_interface = setup_teleop_device(teleoperation_callbacks)
    teleop_interface.add_callback("R", reset_recording_instance)

    # Setup for Mink IK state synchronization (for XR controller with T1/humanoid robots)
    upper_body_joint_ids = None
    robot = None
    if "robot" in env.scene.keys():
        robot = env.scene["robot"]
        # Upper body joint names for T1 humanoid robot (16 joints)
        upper_body_joint_names = [
            "AAHead_yaw", "Head_pitch",
            "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
            "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
            "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
            "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
        ]
        try:
            upper_body_joint_ids = robot.find_joints(upper_body_joint_names, preserve_order=True)[0]
            omni.log.info("Mink IK state synchronization enabled for XR teleoperation")
        except Exception as e:
            omni.log.warn(f"Could not find upper body joints for state sync: {e}. State sync disabled.")
            upper_body_joint_ids = None

    # Reset before starting
    env.sim.reset()
    env.reset()
    teleop_interface.reset()

    # Sync device state on initial reset
    if upper_body_joint_ids is not None and robot is not None:
        measured_joint_pos = robot.data.joint_pos[0, upper_body_joint_ids]
        if hasattr(teleop_interface, 'set_measured_joint_positions'):
            teleop_interface.set_measured_joint_positions(measured_joint_pos)
            omni.log.info("Initial device state synchronized with simulation")
        if hasattr(teleop_interface, '_retargeters'):
            for retargeter in teleop_interface._retargeters:
                if hasattr(retargeter, 'reset'):
                    # Pass joint positions as optional kwarg - retargeters use it if needed
                    retargeter.reset(joint_positions=measured_joint_pos.cpu().numpy())
                    omni.log.info(f"Reset {type(retargeter).__name__} with state sync")

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
    instruction_display = setup_ui(label_text, env)

    subtasks = {}

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # Update device state with measured joint positions (for Mink IK state sync)
            if upper_body_joint_ids is not None and robot is not None:
                measured_joint_pos = robot.data.joint_pos[0, upper_body_joint_ids]
                if hasattr(teleop_interface, 'set_measured_joint_positions'):
                    teleop_interface.set_measured_joint_positions(measured_joint_pos)

            # Get keyboard command
            action = teleop_interface.advance()

            # Check if action is valid (None means tracking data not ready yet)
            if action is None:
                # No valid tracking data yet - just render without stepping
                env.sim.render()
                continue

            # Expand to batch dimension
            actions = action.repeat(env.num_envs, 1)

            # Always step the environment to provide visual feedback
            # Recording state only controls whether the data is saved
            obv = env.step(actions)

            # If not recording, clear the episode data to prevent accumulation
            # This allows the robot to move without recording unwanted data
            if not running_recording_instance:
                # Clear the current episode data for env 0
                env.recorder_manager.get_episode(0).data.clear()

            # Update subtask instructions
            if subtasks is not None:
                if subtasks == {}:
                    subtasks = obv[0].get("subtask_terms")
                elif subtasks:
                    show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)

            # Check for success condition
            success_step_count, success_reset_needed = process_success_condition(env, success_term, success_step_count)
            if success_reset_needed:
                should_reset_recording_instance = True

            # Update demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)

            # Check if we've reached the desired number of demos
            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                instruction_display.show_demo(label_text)
                print(label_text)
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    if rate_limiter:
                        rate_limiter.sleep(env)
                    else:
                        env.sim.render()
                break

            # Handle reset if requested
            if should_reset_recording_instance:
                success_step_count = handle_reset(
                    env, success_step_count, instruction_display, label_text,
                    teleop_interface, robot, upper_body_joint_ids
                )
                should_reset_recording_instance = False

            # Check if simulation is stopped
            if env.sim.is_stopped():
                break

            # Rate limiting
            if rate_limiter:
                rate_limiter.sleep(env)

    return current_recorded_demo_count


def main() -> None:
    """Collect demonstrations from the environment using teleop interfaces.

    Main function that orchestrates the entire process:
    1. Sets up rate limiting based on configuration
    2. Creates output directories for saving demonstrations
    3. Configures the environment
    4. Runs the simulation loop to collect demonstrations
    5. Cleans up resources when done

    Raises:
        Exception: Propagates exceptions from any of the called functions
    """
    # XR controller mode: enable rate limiting using step_hz argument
    rate_limiter = RateLimiter(hz=args_cli.step_hz)

    # Set up output directories
    output_dir, output_file_name = setup_output_directories()

    # Create and configure environment
    global env_cfg  # Make env_cfg available to setup_teleop_device
    env_cfg, success_term = create_environment_config(output_dir, output_file_name)

    # Create environment
    env = create_environment(env_cfg)

    # Run simulation loop
    current_recorded_demo_count = run_simulation_loop(env, None, success_term, rate_limiter)

    # Clean up
    env.close()
    print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
