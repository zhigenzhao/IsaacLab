import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_reached(
    env: "ManagerBasedRLEnv",
    command_name: str,
    radius: float,
    hold_time: float,
    velocity_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """terminate when robot base is within radius of target with low velocity for hold_time duration.

    args:
        env: the environment instance
        command_name: name of the command term containing target points (n, 2)
        radius: distance threshold to consider robot at target
        hold_time: time duration robot must stay within radius with low velocity
        velocity_threshold: maximum base velocity magnitude to start/continue counter
        asset_cfg: scene entity configuration for the robot

    returns:
        boolean tensor (num_envs,) indicating which environments should terminate
    """
    # get robot asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # get target positions from command manager (num_envs, 2)
    target_pos = env.command_manager.get_command(command_name)

    # get robot base position (num_envs, 3) - only use x, y
    robot_pos_xy = asset.data.root_pos_w[:, :2]

    # get robot base linear velocity (num_envs, 3)
    robot_vel = asset.data.root_lin_vel_w
    robot_speed = torch.norm(robot_vel, dim=1)

    # compute distance to target
    distance = torch.norm(robot_pos_xy - target_pos, dim=1)

    # check conditions: within radius AND velocity below threshold
    within_radius = distance < radius
    velocity_low = robot_speed < velocity_threshold
    conditions_met = within_radius & velocity_low

    # track time spent within radius for each environment
    if not hasattr(env, "_target_hold_timer"):
        env._target_hold_timer = torch.zeros(env.num_envs, device=env.device)

    # update timer: increment if conditions met, reset otherwise
    env._target_hold_timer = torch.where(
        conditions_met,
        env._target_hold_timer + env.step_dt,
        torch.zeros_like(env._target_hold_timer)
    )

    # terminate if held for required duration
    return env._target_hold_timer >= hold_time
