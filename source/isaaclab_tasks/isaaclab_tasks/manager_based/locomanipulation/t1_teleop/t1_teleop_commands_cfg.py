from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

import isaaclab.sim as sim_utils
from .t1_teleop_commands import DummyCommand, DestinationPointCommand


@configclass
class DummyCommandCfg(CommandTermCfg):
    """Configuration for DummyCommand."""
    class_type : type = DummyCommand
    resampling_time_range : tuple[float, float] = (1e9, 1e9)
    dim: int = 3


@configclass
class DestinationPointCommandCfg(CommandTermCfg):
    """Configuration for DestinationPointCommand."""
    class_type: type = DestinationPointCommand

    @configclass
    class Ranges:
        """Uniform distribution of ranges."""

        x: tuple[float, float] = MISSING
        """x-pos of the target point."""

        y: tuple[float, float] = MISSING
        """y-pos of the target point."""

    ranges: Ranges = MISSING

    goal_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/destination_goal",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
        },
    )
    """configuration for the goal visualization marker (yellow sphere)."""
    debug_vis = True
