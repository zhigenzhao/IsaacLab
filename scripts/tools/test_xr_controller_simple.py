#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple test for XRoboToolkit XR Controller device - prints and exits immediately."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test XRoboToolkit XR Controller device.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

print("\n" + "=" * 60)
print("XR Controller Simple Test Starting...")
print("=" * 60)

try:
    from isaaclab.devices.xrobotoolkit import XRControllerDevice, XRControllerDeviceCfg
    print("✓ Successfully imported XRControllerDevice")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    simulation_app.close()
    exit(1)

# Create device configuration
print("\nCreating device configuration...")
cfg = XRControllerDeviceCfg(
    pos_sensitivity=0.4,
    rot_sensitivity=0.8,
    control_mode="right_hand",
    gripper_source="trigger",
    sim_device="cpu"
)
print("✓ Configuration created")

# Create device
print("\nCreating XR Controller device...")
try:
    device = XRControllerDevice(cfg)
    print("✓ Device created successfully!")
    print(device)

    # Get one sample of raw data
    print("\nGetting one raw data sample...")
    raw_data = device._get_raw_data()
    print(f"✓ Got raw data with keys: {list(raw_data.keys())}")
    print(f"  Left controller position: {raw_data['left_controller'][:3]}")
    print(f"  Right controller position: {raw_data['right_controller'][:3]}")

    # Test advance
    print("\nTesting advance() method...")
    result = device.advance()
    print(f"✓ advance() returned type: {type(result)}")

    # Cleanup
    del device
    print("\n✓ Device cleaned up")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60 + "\n")

# Close the simulator
simulation_app.close()