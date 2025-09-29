#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script for XRoboToolkit retargeters.

This script tests the XR controller retargeters by:
1. Creating an XR controller device
2. Attaching retargeters to it
3. Reading controller data and displaying retargeted outputs
"""

import argparse
import numpy as np
import time

from isaaclab.devices.xrobotoolkit import (
    XRControllerDevice,
    XRControllerDeviceCfg,
    XRGripperRetargeter,
    XRGripperRetargeterCfg,
    XRSe3AbsRetargeter,
    XRSe3AbsRetargeterCfg,
    XRSe3RelRetargeter,
    XRSe3RelRetargeterCfg,
)


def test_relative_retargeter():
    """Test the relative SE(3) retargeter."""
    print("\n" + "=" * 80)
    print("Testing XRSe3RelRetargeter (Delta-based control)")
    print("=" * 80)
    print("Instructions:")
    print("- Squeeze the right grip button to activate control")
    print("- Move the controller to see delta commands")
    print("- Release grip to deactivate")
    print("- Press Ctrl+C to exit")
    print()

    # Create device
    device_cfg = XRControllerDeviceCfg(
        control_mode="right_hand",
        pos_sensitivity=0.4,
        rot_sensitivity=0.8,
    )
    device = XRControllerDevice(cfg=device_cfg)

    # Create retargeter with visualization
    retargeter_cfg = XRSe3RelRetargeterCfg(
        control_hand="right",
        pos_scale_factor=2.0,
        rot_scale_factor=2.0,
        activation_source="grip",
        activation_threshold=0.9,
        alpha_pos=0.5,
        alpha_rot=0.5,
        zero_out_xy_rotation=False,
        enable_visualization=False,  # Set to True if in Isaac Sim
        sim_device="cpu",
    )
    retargeter = XRSe3RelRetargeter(cfg=retargeter_cfg)

    try:
        while True:
            # Get raw data from device
            raw_data = device._get_raw_data()

            # Retarget the data
            delta_command = retargeter.retarget(raw_data)

            # Display results
            grip_value = raw_data.get("right_grip", 0.0)
            print(f"\rGrip: {grip_value:.2f} | Delta: [{delta_command[0]:+.3f}, {delta_command[1]:+.3f}, {delta_command[2]:+.3f}] "
                  f"[{delta_command[3]:+.3f}, {delta_command[4]:+.3f}, {delta_command[5]:+.3f}]", end="")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nTest stopped by user")


def test_absolute_retargeter():
    """Test the absolute SE(3) retargeter."""
    print("\n" + "=" * 80)
    print("Testing XRSe3AbsRetargeter (Absolute control)")
    print("=" * 80)
    print("Instructions:")
    print("- Move the right controller to see absolute pose commands")
    print("- Press Ctrl+C to exit")
    print()

    # Create device
    device_cfg = XRControllerDeviceCfg(
        control_mode="right_hand",
        pos_sensitivity=0.4,
        rot_sensitivity=0.8,
    )
    device = XRControllerDevice(cfg=device_cfg)

    # Create retargeter
    retargeter_cfg = XRSe3AbsRetargeterCfg(
        control_hand="right",
        zero_out_xy_rotation=False,
        enable_visualization=False,  # Set to True if in Isaac Sim
        sim_device="cpu",
    )
    retargeter = XRSe3AbsRetargeter(cfg=retargeter_cfg)

    try:
        while True:
            # Get raw data from device
            raw_data = device._get_raw_data()

            # Retarget the data
            abs_command = retargeter.retarget(raw_data)

            # Display results
            print(f"\rPos: [{abs_command[0]:+.3f}, {abs_command[1]:+.3f}, {abs_command[2]:+.3f}] "
                  f"Quat: [{abs_command[3]:+.3f}, {abs_command[4]:+.3f}, {abs_command[5]:+.3f}, {abs_command[6]:+.3f}]", end="")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nTest stopped by user")


def test_gripper_retargeter():
    """Test the gripper retargeter."""
    print("\n" + "=" * 80)
    print("Testing XRGripperRetargeter")
    print("=" * 80)
    print("Instructions:")
    print("- Squeeze the right trigger to control gripper")
    print("- Watch the gripper command value")
    print("- Press Ctrl+C to exit")
    print()

    # Create device
    device_cfg = XRControllerDeviceCfg(
        control_mode="right_hand",
    )
    device = XRControllerDevice(cfg=device_cfg)

    # Test continuous mode
    print("\n--- Testing Continuous Mode ---")
    retargeter_cfg = XRGripperRetargeterCfg(
        control_hand="right",
        input_source="trigger",
        mode="continuous",
        sim_device="cpu",
    )
    retargeter = XRGripperRetargeter(cfg=retargeter_cfg)

    try:
        for _ in range(100):
            # Get raw data from device
            raw_data = device._get_raw_data()

            # Retarget the data
            gripper_command = retargeter.retarget(raw_data)

            # Display results
            trigger_value = raw_data.get("right_trigger", 0.0)
            print(f"\rTrigger: {trigger_value:.3f} | Gripper Command: {gripper_command[0]:.3f}", end="")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nTest stopped by user")

    # Test binary mode
    print("\n\n--- Testing Binary Mode ---")
    retargeter_cfg = XRGripperRetargeterCfg(
        control_hand="right",
        input_source="trigger",
        mode="binary",
        binary_threshold=0.5,
        hysteresis=0.1,
        open_value=1.0,
        closed_value=0.0,
        sim_device="cpu",
    )
    retargeter = XRGripperRetargeter(cfg=retargeter_cfg)

    try:
        while True:
            # Get raw data from device
            raw_data = device._get_raw_data()

            # Retarget the data
            gripper_command = retargeter.retarget(raw_data)

            # Display results
            trigger_value = raw_data.get("right_trigger", 0.0)
            state = "OPEN" if gripper_command[0] > 0.5 else "CLOSED"
            print(f"\rTrigger: {trigger_value:.3f} | Gripper: {state} ({gripper_command[0]:.1f})", end="")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nTest stopped by user")


def test_combined():
    """Test all retargeters together."""
    print("\n" + "=" * 80)
    print("Testing Combined Retargeters (SE3 Rel + Gripper)")
    print("=" * 80)
    print("Instructions:")
    print("- Squeeze right grip to activate SE3 control")
    print("- Move controller for delta commands")
    print("- Squeeze right trigger for gripper control")
    print("- Press Ctrl+C to exit")
    print()

    # Create device with retargeters
    device_cfg = XRControllerDeviceCfg(
        control_mode="right_hand",
        retargeters=[
            XRSe3RelRetargeterCfg(
                control_hand="right",
                pos_scale_factor=2.0,
                rot_scale_factor=2.0,
                activation_source="grip",
                activation_threshold=0.9,
                sim_device="cpu",
            ),
            XRGripperRetargeterCfg(
                control_hand="right",
                input_source="trigger",
                mode="continuous",
                sim_device="cpu",
            ),
        ]
    )
    device = XRControllerDevice(cfg=device_cfg)

    try:
        while True:
            # Get combined output using advance()
            commands = device.advance()

            # Display results
            # commands is concatenated: [delta_pos (3), delta_rot (3), gripper (1)]
            grip_value = commands[-1]
            print(f"\rDelta: [{commands[0]:+.3f}, {commands[1]:+.3f}, {commands[2]:+.3f}] "
                  f"[{commands[3]:+.3f}, {commands[4]:+.3f}, {commands[5]:+.3f}] | Gripper: {grip_value:.3f}", end="")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nTest stopped by user")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test XRoboToolkit retargeters")
    parser.add_argument(
        "--mode",
        type=str,
        default="relative",
        choices=["relative", "absolute", "gripper", "combined"],
        help="Which retargeter to test",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("XRoboToolkit Retargeter Test")
    print("=" * 80)

    if args.mode == "relative":
        test_relative_retargeter()
    elif args.mode == "absolute":
        test_absolute_retargeter()
    elif args.mode == "gripper":
        test_gripper_retargeter()
    elif args.mode == "combined":
        test_combined()


if __name__ == "__main__":
    main()