# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import torch
from typing import Dict, Optional

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodied_challenge.managers.events import visualize_rigid_body_pose
from embodichain.utils import logger

from embodichain.lab.gym.envs.tasks.tableware.base_agent_env import BaseAgentEnv
from .action_bank import BeakerMixerActionBank

__all__ = ["BeakerMixerEnv", "BeakerMixerAgentEnv"]


@register_env("BeakerMixer-v0", max_episode_steps=600)
class BeakerMixerEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)
        debug_render_cfg = {}
        if isinstance(action_config, dict):
            debug_render_cfg = action_config.pop("debug_render", {})

        if action_config is not None:
            self.action_config = action_config

    def create_demo_action_list(self, *args, **kwargs):
        """Create a demonstration action list for BeakerMixer task."""
        logger.log_info("Create demo action list for BeakerMixerTask.")
        action_list = None

        if getattr(self, "action_config") is not None:
            self._init_action_bank(BeakerMixerActionBank, self.action_config)
            action_list = self.create_expert_demo_action_list(*args, **kwargs)
        else:
            logger.log_error("No action_config found in env, please check again.")

        if action_list is None:
            return action_list

        logger.log_info(
            f"Demo action list created with {len(action_list)} steps.", color="green"
        )
        return action_list

    def create_expert_demo_action_list(self, **kwargs):
        """Create expert demonstration actions from action bank graph."""
        if hasattr(self, "action_bank") is False or self.action_bank is None:
            logger.log_error(
                "Action bank is not initialized. Cannot create expert demo action list."
            )

        ret = self.action_bank.create_action_list(
            self, self.graph_compose, self.packages
        )

        if ret is None:
            logger.log_warning("Failed to generate expert demo action list.")
            return None

        left_arm_joints = self.robot.get_joint_ids(name="left_arm", remove_mimic=True)
        right_arm_joints = self.robot.get_joint_ids(
            name="right_arm", remove_mimic=True
        )
        left_eef_joints = self.robot.get_joint_ids(name="left_eef", remove_mimic=True)
        right_eef_joints = self.robot.get_joint_ids(
            name="right_eef", remove_mimic=True
        )

        total_traj_num = ret[list(ret.keys())[0]].shape[-1]
        num_active_joints = len(self.active_joint_ids)
        actions = torch.zeros(
            (total_traj_num, self.num_envs, num_active_joints), dtype=torch.float32
        )

        global_to_active_idx = {
            joint_id: active_idx
            for active_idx, joint_id in enumerate(self.active_joint_ids)
        }

        for key, joints in [
            ("left_arm", left_arm_joints),
            ("left_eef", left_eef_joints),
            ("right_arm", right_arm_joints),
            ("right_eef", right_eef_joints),
        ]:
            if key in ret:
                # TODO: only 1 env supported now
                local_action_data = torch.as_tensor(ret[key].T, dtype=torch.float32)
                for i, joint_id in enumerate(joints):
                    if joint_id in global_to_active_idx:
                        active_idx = global_to_active_idx[joint_id]
                        actions[:, 0, active_idx] = local_action_data[:, i]
        return actions

    def compute_task_state(self, **kwargs):
        beaker = self.sim.get_rigid_object("beaker")
        mixer = self.sim.get_rigid_object("beaker_mixer")

        beaker_pose = beaker.get_local_pose(to_matrix=True)
        mixer_pose = mixer.get_local_pose(to_matrix=True)

        beaker_fall = self._is_fall(beaker_pose)
        success = torch.zeros_like(beaker_fall, dtype=torch.bool)

        return success, beaker_fall, {}

    def is_task_success(self, **kwargs) -> torch.Tensor:

        beaker = self.sim.get_rigid_object("beaker")
        beaker_mixer = self.sim.get_rigid_object("beaker_mixer")

        beaker_final_xpos = beaker.get_local_pose(to_matrix=True)
        beaker_mixer_final_xpos = beaker_mixer.get_local_pose(to_matrix=True)

        beaker_ret = self._is_fall(beaker_final_xpos)
        beaker_pos_xy = beaker_final_xpos[:, :2, 3]
        beaker_mixer_pos_xy = beaker_mixer_final_xpos[:, :2, 3]

        # Success requires the beaker to stay near the mixer in XY plane.
        beaker_mixer_dist = torch.linalg.norm(beaker_pos_xy - beaker_mixer_pos_xy, dim=-1)
        # print(f"Beaker-Mixer distance: {beaker_mixer_dist.item():.4f}")
        dist_threshold = 0.08
        beaker_near_mixer = beaker_mixer_dist <= dist_threshold

        return (~beaker_ret) & beaker_near_mixer

    def _is_fall(self, pose: torch.Tensor) -> torch.Tensor:
        # Extract z-axis from rotation matrix (last column, first 3 elements)
        pose_rz = pose[:, :3, 2]
        world_z_axis = torch.tensor([0, 0, 1], dtype=pose.dtype, device=pose.device)

        # Compute dot product for each batch element
        dot_product = torch.sum(pose_rz * world_z_axis, dim=-1)  # Shape: (batch_size,)

        # Clamp to avoid numerical issues with arccos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        # Compute angle and check if fallen
        angle = torch.arccos(dot_product)
        return angle >= torch.pi / 4

@register_env("BeakerMixerAgent-v0", max_episode_steps=600)
class BeakerMixerAgentEnv(BaseAgentEnv, BeakerMixerEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        super()._init_agents(**kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        return obs, info
