import torch
from isaaclab.managers import CommandTerm
from collections.abc import Sequence
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from dataclasses import MISSING
import isaaclab.sim as sim_utils


class DummyCommand(CommandTerm):
    """
    A dummy command that can be externally controlled.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._external_command = torch.zeros(self.num_envs, self.cfg.dim, device=self.device)

    def set_command(self, command: torch.Tensor) -> None:
        """Set the command externally."""
        if command.dim() == 1:
            self._external_command[:] = command
        else:
            self._external_command.copy_(command)

    def _update_command(self) -> None:
        """Required by CommandTerm but does nothing."""
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Required by CommandTerm. Reset to zero."""
        self._external_command[env_ids] = 0.0

    def _update_metrics(self):
        pass

    def compute(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return the external command."""
        return self._external_command

    @property
    def command(self) -> torch.Tensor:
        return self._external_command


class DestinationPointCommand(CommandTerm):
    """
    A destination point for the target.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.destination_point = torch.zeros(self._env.num_envs, 2, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The command tensor. Shape is (num_envs, 2)."""
        return self.destination_point

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the destination point for the specified environments."""
        # sample x and y coordinates uniformly from the ranges
        self.destination_point[env_ids, 0] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(*self.cfg.ranges.x)
        self.destination_point[env_ids, 1] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(*self.cfg.ranges.y)

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects."""
        # set visibility of goal visualizer
        if not hasattr(self, "goal_visualizer"):
            self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            self.goal_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        """Callback for debug visualization."""
        # create marker positions (x, y, z=0) for 2D destination points
        marker_positions = torch.zeros((self.destination_point.shape[0], 3))
        marker_positions[:, :2] = self.destination_point
        marker_positions[:, 2] = 1.5

        # visualize the goal positions
        self.goal_visualizer.visualize(marker_positions)
