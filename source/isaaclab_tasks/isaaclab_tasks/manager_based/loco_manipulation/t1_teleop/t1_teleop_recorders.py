# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
from dataclasses import MISSING

import torch
from collections.abc import Sequence

from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass
from isaaclab.managers import RecorderTermCfg


class InitialStateRecorder(RecorderTerm):
    """Recorder term that records the initial state of the environment after reset."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        def extract_env_ids_values(value):
            nonlocal env_ids
            if isinstance(value, dict):
                return {k: extract_env_ids_values(v) for k, v in value.items()}
            return value[env_ids]

        return "initial_state", extract_env_ids_values(self._env.scene.get_state(is_relative=True))


class PostStepStatesRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_post_step(self):
        return "states", self._env.scene.get_state(is_relative=True)


class PreStepActionsRecorder(RecorderTerm):
    """Recorder term that records the actions in the beginning of each step."""

    def record_pre_step(self):
        return "actions", self._env.action_manager.action


class PreStepFlatPolicyObservationsRecorder(RecorderTerm):
    """Recorder term that records the policy group observations in each step."""

    def record_pre_step(self):
        return "obs", self._env.obs_buf["policy"]


class PostStepProcessedActionsRecorder(RecorderTerm):
    """Recorder term that records processed actions at the end of each step."""

    def record_post_step(self):
        processed_actions = None

        # Loop through active terms and concatenate their processed actions
        for term_name in self._env.action_manager.active_terms:
            term_actions = self._env.action_manager.get_term(term_name).processed_actions.clone()
            if processed_actions is None:
                processed_actions = term_actions
            else:
                processed_actions = torch.cat([processed_actions, term_actions], dim=-1)

        return "processed_actions", processed_actions


class PostStepCommandRecorder(RecorderTerm):
    """Recorder term that records processed commands at the end of each step."""

    def record_post_step(self):
        res = {}
        self._env: ManagerBasedRLEnv
        for term in self.cfg.recording_terms:
            res[term] = self._env.command_manager.get_command(term)
        return "commands", res


@configclass
class PostStepCommandRecorderCfg(RecorderTermCfg):
    """Configuration for the post step processed actions recorder term."""
    class_type: type[RecorderTerm] = PostStepCommandRecorder
    recording_terms: list[str] = MISSING
