#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to copy physical parameters from one USD file to another while preserving joint axes."""

import argparse
import shutil
from pathlib import Path
from pxr import Usd, UsdPhysics, Sdf


def copy_physics_parameters(source_usd: Path, target_usd: Path, output_usd: Path, preserve_axes: bool = True):
    """
    Copy physics parameters from source USD to target USD.

    Args:
        source_usd: Path to source USD file (parameters to copy from)
        target_usd: Path to target USD file (base structure)
        output_usd: Path to output USD file
        preserve_axes: If True, preserve joint axes from target file
    """

    # Create a copy of the target file
    print(f"Creating copy of target file: {target_usd} -> {output_usd}")
    shutil.copy2(target_usd, output_usd)

    # Open both stages
    print(f"Opening source file: {source_usd}")
    source_stage = Usd.Stage.Open(str(source_usd))
    if not source_stage:
        raise RuntimeError(f"Failed to open source USD file: {source_usd}")

    print(f"Opening output file for editing: {output_usd}")
    output_stage = Usd.Stage.Open(str(output_usd))
    if not output_stage:
        raise RuntimeError(f"Failed to open output USD file: {output_usd}")

    # Find all joints in source
    source_joints = {}
    for prim in source_stage.Traverse():
        if "Joint" in prim.GetTypeName():
            source_joints[prim.GetName()] = prim

    print(f"\nFound {len(source_joints)} joints in source file")

    # Parameters to copy
    physics_params_to_copy = [
        "drive:angular:physics:damping",
        "drive:angular:physics:stiffness",
        "drive:angular:physics:maxForce",
        "physics:lowerLimit",
        "physics:upperLimit",
    ]

    # Parameters to preserve (not copy)
    params_to_preserve = [
        "physics:axis",
        "physics:localRot0",
        "physics:localRot1",
    ] if preserve_axes else []

    copied_count = 0
    skipped_count = 0

    # Traverse output stage and update matching joints
    for output_prim in output_stage.Traverse():
        if "Joint" in output_prim.GetTypeName():
            joint_name = output_prim.GetName()

            # Check if this joint exists in source
            if joint_name in source_joints:
                source_prim = source_joints[joint_name]

                print(f"\nProcessing joint: {joint_name}")

                # Store axis parameters if preserving
                preserved_values = {}
                if preserve_axes:
                    for param_name in params_to_preserve:
                        attr = output_prim.GetAttribute(param_name)
                        if attr and attr.Get() is not None:
                            preserved_values[param_name] = attr.Get()
                            print(f"  Preserving {param_name}: {preserved_values[param_name]}")

                # Copy physics parameters
                for param_name in physics_params_to_copy:
                    source_attr = source_prim.GetAttribute(param_name)
                    if source_attr and source_attr.Get() is not None:
                        source_value = source_attr.Get()

                        # Get or create attribute in output
                        output_attr = output_prim.GetAttribute(param_name)
                        if not output_attr:
                            # Create attribute if it doesn't exist
                            attr_type = source_attr.GetTypeName()
                            output_attr = output_prim.CreateAttribute(param_name, attr_type)

                        # Set the value
                        output_attr.Set(source_value)
                        print(f"  Copied {param_name}: {source_value}")
                        copied_count += 1

                # Restore preserved parameters
                if preserve_axes:
                    for param_name, value in preserved_values.items():
                        attr = output_prim.GetAttribute(param_name)
                        if attr:
                            attr.Set(value)
                            print(f"  Restored {param_name}: {value}")
            else:
                print(f"\nSkipping joint (not in source): {joint_name}")
                skipped_count += 1

    # Save the output stage
    print(f"\n\nSaving output file: {output_usd}")
    output_stage.Save()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Source file: {source_usd}")
    print(f"Target file: {target_usd}")
    print(f"Output file: {output_usd}")
    print(f"Total parameters copied: {copied_count}")
    print(f"Joints skipped (not in source): {skipped_count}")
    print(f"Axes preserved: {preserve_axes}")
    print("="*80)

    return output_stage


def main():
    """Main function to copy physics parameters between USD files."""
    parser = argparse.ArgumentParser(
        description="Copy physics parameters from one USD file to another while optionally preserving joint axes."
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path to source USD file (physics parameters to copy FROM)"
    )
    parser.add_argument(
        "target",
        type=str,
        help="Path to target USD file (base structure to copy TO)"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to output USD file (modified target)"
    )
    parser.add_argument(
        "--copy-axes",
        action="store_true",
        help="Copy joint axes from source (default: preserve target axes)"
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    target_path = Path(args.target)
    output_path = Path(args.output)

    # Validate input files
    if not source_path.exists():
        print(f"Error: Source file does not exist: {source_path}")
        return

    if not target_path.exists():
        print(f"Error: Target file does not exist: {target_path}")
        return

    # Check if output file already exists
    if output_path.exists():
        response = input(f"Output file already exists: {output_path}\nOverwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy physics parameters
    preserve_axes = not args.copy_axes

    try:
        copy_physics_parameters(
            source_usd=source_path,
            target_usd=target_path,
            output_usd=output_path,
            preserve_axes=preserve_axes
        )

        print("\n✓ Physics parameters copied successfully!")

    except Exception as e:
        print(f"\n✗ Error copying physics parameters: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
