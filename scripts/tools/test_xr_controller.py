#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test script for XRoboToolkit XR Controller device.

This script demonstrates how to use the XRControllerDevice for teleoperation.
It creates the device and prints the raw data output to verify the connection
and data flow from the XR controllers.

Usage:
    python test_xr_controller.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test XRoboToolkit XR Controller device.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator in headless mode for testing
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import torch

try:
    from isaaclab.devices.xrobotoolkit import XRControllerDevice, XRControllerDeviceCfg
except ImportError:
    print("Error: XRoboToolkit device not available. Make sure xrobotoolkit_sdk is installed.")
    simulation_app.close()
    exit(1)


def main():
    """Test the XR Controller device."""
    print("\n" + "=" * 60)
    print("Testing XRoboToolkit XR Controller Device")
    print("=" * 60 + "\n")

    # Create device configuration
    cfg = XRControllerDeviceCfg(
        pos_sensitivity=0.4,
        rot_sensitivity=0.8,
        control_mode="right_hand",
        gripper_source="trigger",
        deadzone_threshold=0.05,
        sim_device="cpu"
    )

    try:
        # Create device
        print("Creating XR Controller device...")
        device = XRControllerDevice(cfg)
        print("✓ Device created successfully\n")
        print(device)

        # Test callback system
        def test_callback():
            print("  🔄 Test callback triggered!")

        device.add_callback("RESET", test_callback)
        print("\n✓ Callback registered")

        # Test raw data collection
        print("\nTesting raw data collection (collecting 10 samples)...")
        for i in range(10):
            raw_data = device._get_raw_data()

            if i % 2 == 0:  # Print every other sample
                print(f"\nSample {i + 1}:")
                print(f"  Left Controller:  {raw_data['left_controller'][:3]}")
                print(f"  Right Controller: {raw_data['right_controller'][:3]}")
                print(f"  Left Trigger:  {raw_data['left_trigger']:.3f}")
                print(f"  Right Trigger: {raw_data['right_trigger']:.3f}")
                print(f"  Timestamp: {raw_data['timestamp']}")

            simulation_app.update()
            time.sleep(0.1)

        print(f"\n✓ Collected 10 data samples successfully")

        # Test advance method
        print("\nTesting advance() method...")
        result = device.advance()
        print(f"✓ advance() returned: {type(result)}")
        if isinstance(result, dict):
            print(f"  Data keys: {list(result.keys())}")
        elif isinstance(result, torch.Tensor):
            print(f"  Tensor shape: {result.shape}")

        # Clean up
        del device
        print("\n✓ Device cleaned up")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("XR Controller device testing PASSED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Verify controller tracking with XRoboToolkit PC Service running")
    print("2. Test button callbacks by pressing controller buttons")
    print("3. Develop retargeters to convert raw data to robot commands")
    print("4. Integrate with IsaacLab environments for teleoperation")
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        simulation_app.close()