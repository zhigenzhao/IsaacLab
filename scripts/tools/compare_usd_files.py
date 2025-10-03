#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to compare two USD files and analyze their structure and physical parameters."""

import argparse
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade


def get_prim_info(prim):
    """Extract comprehensive information from a USD prim."""
    info = {
        "type": prim.GetTypeName(),
        "path": str(prim.GetPath()),
        "children": [child.GetName() for child in prim.GetChildren()],
        "attributes": {},
        "relationships": {},
    }

    # Get all attributes
    for attr in prim.GetAttributes():
        attr_name = attr.GetName()
        try:
            value = attr.Get()
            info["attributes"][attr_name] = str(value) if value is not None else "None"
        except:
            info["attributes"][attr_name] = "<unable to read>"

    # Get all relationships
    for rel in prim.GetRelationships():
        rel_name = rel.GetName()
        targets = rel.GetTargets()
        info["relationships"][rel_name] = [str(t) for t in targets]

    return info


def get_joint_info(stage, prim):
    """Extract joint-specific information."""
    joint_info = {
        "name": prim.GetName(),
        "type": prim.GetTypeName(),
        "path": str(prim.GetPath()),
    }

    # Get all physics-related attributes directly
    drive_attrs = {}
    for attr in prim.GetAttributes():
        attr_name = attr.GetName()
        if "drive" in attr_name.lower() or "physics:" in attr_name:
            try:
                value = attr.Get()
                drive_attrs[attr_name] = value
            except:
                pass

    if drive_attrs:
        joint_info["drive_attributes"] = drive_attrs

    # Joint limits
    for attr in prim.GetAttributes():
        attr_name = attr.GetName()
        if "limit" in attr_name.lower() or "upper" in attr_name.lower() or "lower" in attr_name.lower():
            joint_info[attr_name] = attr.Get()

    # Body relationships
    body0_rel = prim.GetRelationship("physics:body0")
    body1_rel = prim.GetRelationship("physics:body1")
    if body0_rel:
        joint_info["body0"] = [str(t) for t in body0_rel.GetTargets()]
    if body1_rel:
        joint_info["body1"] = [str(t) for t in body1_rel.GetTargets()]

    return joint_info


def get_rigid_body_info(prim):
    """Extract rigid body physics information."""
    body_info = {
        "name": prim.GetName(),
        "path": str(prim.GetPath()),
    }

    # Get all physics-related attributes
    for attr in prim.GetAttributes():
        attr_name = attr.GetName()
        if "physics:" in attr_name or "mass" in attr_name.lower() or "inertia" in attr_name.lower():
            try:
                value = attr.Get()
                body_info[attr_name] = value
            except:
                pass

    return body_info


def analyze_usd_structure(usd_path: Path):
    """Analyze USD file structure and extract key information."""
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_path}")

    analysis = {
        "file_path": str(usd_path),
        "joints": {},
        "rigid_bodies": {},
        "links": [],
        "articulation_root": None,
        "hierarchy": {},
    }

    # Find articulation root
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            analysis["articulation_root"] = str(prim.GetPath())
            break

    # Traverse and categorize prims
    for prim in stage.Traverse():
        prim_type = prim.GetTypeName()

        # Joints
        if "Joint" in prim_type:
            joint_info = get_joint_info(stage, prim)
            analysis["joints"][prim.GetName()] = joint_info

        # Rigid bodies (links)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_info = get_rigid_body_info(prim)
            analysis["rigid_bodies"][prim.GetName()] = body_info
            analysis["links"].append(prim.GetName())

        # Build hierarchy
        if prim_type in ["Xform", "Mesh", "Scope"]:
            parent_path = str(prim.GetPath().GetParentPath())
            if parent_path not in analysis["hierarchy"]:
                analysis["hierarchy"][parent_path] = []
            analysis["hierarchy"][parent_path].append({
                "name": prim.GetName(),
                "type": prim_type,
                "path": str(prim.GetPath()),
            })

    return analysis


def compare_structures(analysis1, analysis2):
    """Compare two USD structure analyses."""
    comparison = {
        "file1": analysis1["file_path"],
        "file2": analysis2["file_path"],
        "joint_comparison": {},
        "rigid_body_comparison": {},
        "summary": {},
    }

    # Compare joint counts
    joints1 = set(analysis1["joints"].keys())
    joints2 = set(analysis2["joints"].keys())

    comparison["summary"]["joints_only_in_file1"] = list(joints1 - joints2)
    comparison["summary"]["joints_only_in_file2"] = list(joints2 - joints1)
    comparison["summary"]["common_joints"] = list(joints1 & joints2)
    comparison["summary"]["joint_count_file1"] = len(joints1)
    comparison["summary"]["joint_count_file2"] = len(joints2)

    # Compare rigid bodies
    bodies1 = set(analysis1["rigid_bodies"].keys())
    bodies2 = set(analysis2["rigid_bodies"].keys())

    comparison["summary"]["bodies_only_in_file1"] = list(bodies1 - bodies2)
    comparison["summary"]["bodies_only_in_file2"] = list(bodies2 - bodies1)
    comparison["summary"]["common_bodies"] = list(bodies1 & bodies2)
    comparison["summary"]["body_count_file1"] = len(bodies1)
    comparison["summary"]["body_count_file2"] = len(bodies2)

    # Compare common joints
    for joint_name in comparison["summary"]["common_joints"]:
        joint1 = analysis1["joints"][joint_name]
        joint2 = analysis2["joints"][joint_name]

        differences = {}

        # Compare drive properties if they exist
        if "drive_attributes" in joint1 and "drive_attributes" in joint2:
            for key in joint1["drive_attributes"]:
                if key in joint2["drive_attributes"]:
                    if joint1["drive_attributes"][key] != joint2["drive_attributes"][key]:
                        differences[key] = {
                            "file1": joint1["drive_attributes"][key],
                            "file2": joint2["drive_attributes"][key],
                        }

        # Compare bodies
        if joint1.get("body0") != joint2.get("body0"):
            differences["body0"] = {
                "file1": joint1.get("body0"),
                "file2": joint2.get("body0"),
            }
        if joint1.get("body1") != joint2.get("body1"):
            differences["body1"] = {
                "file1": joint1.get("body1"),
                "file2": joint2.get("body1"),
            }

        if differences:
            comparison["joint_comparison"][joint_name] = differences

    # Compare common rigid bodies
    for body_name in comparison["summary"]["common_bodies"]:
        body1 = analysis1["rigid_bodies"][body_name]
        body2 = analysis2["rigid_bodies"][body_name]

        differences = {}
        for key in ["mass", "density", "center_of_mass", "inertia", "collision_enabled"]:
            if key in body1 and key in body2:
                if body1[key] != body2[key]:
                    differences[key] = {
                        "file1": body1[key],
                        "file2": body2[key],
                    }

        if differences:
            comparison["rigid_body_comparison"][body_name] = differences

    return comparison


def print_comparison(comparison):
    """Pretty print the comparison results."""
    print("\n" + "="*80)
    print("USD FILE COMPARISON")
    print("="*80)

    print(f"\nFile 1: {comparison['file1']}")
    print(f"File 2: {comparison['file2']}")

    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)

    summary = comparison["summary"]

    print(f"\nJoint Count:")
    print(f"  File 1: {summary['joint_count_file1']}")
    print(f"  File 2: {summary['joint_count_file2']}")

    if summary["joints_only_in_file1"]:
        print(f"\nJoints only in File 1 ({len(summary['joints_only_in_file1'])}):")
        for joint in summary["joints_only_in_file1"]:
            print(f"  - {joint}")

    if summary["joints_only_in_file2"]:
        print(f"\nJoints only in File 2 ({len(summary['joints_only_in_file2'])}):")
        for joint in summary["joints_only_in_file2"]:
            print(f"  - {joint}")

    print(f"\nRigid Body Count:")
    print(f"  File 1: {summary['body_count_file1']}")
    print(f"  File 2: {summary['body_count_file2']}")

    if summary["bodies_only_in_file1"]:
        print(f"\nBodies only in File 1 ({len(summary['bodies_only_in_file1'])}):")
        for body in summary["bodies_only_in_file1"]:
            print(f"  - {body}")

    if summary["bodies_only_in_file2"]:
        print(f"\nBodies only in File 2 ({len(summary['bodies_only_in_file2'])}):")
        for body in summary["bodies_only_in_file2"]:
            print(f"  - {body}")

    if comparison["joint_comparison"]:
        print("\n" + "-"*80)
        print("JOINT DIFFERENCES (Common Joints)")
        print("-"*80)

        for joint_name, differences in comparison["joint_comparison"].items():
            print(f"\nJoint: {joint_name}")
            for key, values in differences.items():
                print(f"  {key}:")
                print(f"    File 1: {values['file1']}")
                print(f"    File 2: {values['file2']}")
    else:
        print("\nNo differences found in common joints.")

    if comparison["rigid_body_comparison"]:
        print("\n" + "-"*80)
        print("RIGID BODY DIFFERENCES (Common Bodies)")
        print("-"*80)

        for body_name, differences in comparison["rigid_body_comparison"].items():
            print(f"\nBody: {body_name}")
            for key, values in differences.items():
                print(f"  {key}:")
                print(f"    File 1: {values['file1']}")
                print(f"    File 2: {values['file2']}")
    else:
        print("\nNo differences found in common rigid bodies.")

    print("\n" + "="*80)


def main():
    """Main function to compare two USD files."""
    parser = argparse.ArgumentParser(description="Compare two USD files and analyze their structure.")
    parser.add_argument("file1", type=str, help="Path to first USD file")
    parser.add_argument("file2", type=str, help="Path to second USD file")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis of each file")

    args = parser.parse_args()

    file1 = Path(args.file1)
    file2 = Path(args.file2)

    if not file1.exists():
        print(f"Error: File 1 does not exist: {file1}")
        return

    if not file2.exists():
        print(f"Error: File 2 does not exist: {file2}")
        return

    print("Analyzing USD files...")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")

    # Analyze both files
    analysis1 = analyze_usd_structure(file1)
    analysis2 = analyze_usd_structure(file2)

    if args.detailed:
        print("\n" + "="*80)
        print("DETAILED ANALYSIS - FILE 1")
        print("="*80)
        print(f"\nArticulation Root: {analysis1['articulation_root']}")
        print(f"\nJoints ({len(analysis1['joints'])}):")
        for name, info in analysis1["joints"].items():
            print(f"  - {name}: {info['type']}")
            if "drive" in info:
                print(f"    Drive: {info['drive']}")

        print(f"\nRigid Bodies ({len(analysis1['rigid_bodies'])}):")
        for name, info in analysis1["rigid_bodies"].items():
            print(f"  - {name}")
            if "mass" in info:
                print(f"    Mass: {info['mass']}")

        print("\n" + "="*80)
        print("DETAILED ANALYSIS - FILE 2")
        print("="*80)
        print(f"\nArticulation Root: {analysis2['articulation_root']}")
        print(f"\nJoints ({len(analysis2['joints'])}):")
        for name, info in analysis2["joints"].items():
            print(f"  - {name}: {info['type']}")
            if "drive" in info:
                print(f"    Drive: {info['drive']}")

        print(f"\nRigid Bodies ({len(analysis2['rigid_bodies'])}):")
        for name, info in analysis2["rigid_bodies"].items():
            print(f"  - {name}")
            if "mass" in info:
                print(f"    Mass: {info['mass']}")

    # Compare structures
    comparison = compare_structures(analysis1, analysis2)
    print_comparison(comparison)


if __name__ == "__main__":
    main()
