from ..dp_teleop_env_cfg import T1DPTeleopEnvCfg, CommandCfg, TerminationCfg
from ..dp_teleop_commands_cfg import DestinationPointCommandCfg
from ..dp_teleop_terminations import target_reached
from isaaclab.managers import TerminationTermCfg as DoneTerm


class WalkToTargetCommandsCfg(CommandCfg):
    destination_point = DestinationPointCommandCfg(
        resampling_time_range=(20.0, 40.0),
        ranges=DestinationPointCommandCfg.Ranges(
            x=(-5.0, 5.0),
            y=(-5.0, 5.0)
        ),
        debug_vis=True
    )


class WalkToTargetTerminationCfg(TerminationCfg):
    reached_target = DoneTerm(
        target_reached,
        params={
            "command_name": "destination_point",
            "radius": 0.35,
            "hold_time": 2.0,
            "velocity_threshold": 0.30
        },
        time_out=True
    )


class T1DpTeleopEnvCfg_WalkToTarget(T1DPTeleopEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.commands = WalkToTargetCommandsCfg()
        self.terminations = WalkToTargetTerminationCfg()
        self.recorders.record_post_step_commands.recording_terms = ["base_velocity", "destination_point"]
