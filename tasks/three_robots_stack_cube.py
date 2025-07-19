from typing import Any, Dict, Tuple

import numpy as np
import sapien
import torch
import math
import yaml
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
# from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
import utils.scenes


@register_env("ThreeRobotsStackCube-rf", max_episode_steps=800)
class ThreeRobotsStackCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("panda", "panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda, Panda]]

    goal_radius = 0.12

    def __init__(
        self,
        *args,
        robot_uids=("panda", "panda", "panda"),
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        assert 'config' in kwargs
        with open(kwargs['config'], 'r', encoding='utf-8') as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        del kwargs['config']
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _default_sensor_configs(self):
        camera_cfg = self.cfg.get('cameras', {})
        sensor_cfg = camera_cfg.get('sensor', [])
        all_camera_configs =[]
        for sensor in sensor_cfg:
            pose = sensor['pose']
            if pose['type'] == 'pose':
                sensor['pose'] = sapien.Pose(*pose['params'])
            elif pose['type'] == 'look_at':
                sensor['pose'] = sapien_utils.look_at(*pose['params'])
            all_camera_configs.append(CameraConfig(**sensor))
        return all_camera_configs

    @property
    def _default_human_render_camera_configs(self):
        camera_cfg = self.cfg.get('cameras', {})
        render_cfg = camera_cfg.get('human_render', [])
        all_camera_configs =[]
        for render in render_cfg:
            pose = render['pose']
            if pose['type'] == 'pose':
                render['pose'] = sapien.Pose(*pose['params'])
            elif pose['type'] == 'look_at':
                render['pose'] = sapien_utils.look_at(*pose['params'])
            all_camera_configs.append(CameraConfig(**render))
        return all_camera_configs

    def _load_agent(self, options: dict):
        init_poses = []
        for agent_cfg in self.cfg['agents']:
            init_poses.append(sapien.Pose(p=agent_cfg['pos']['ppos']['p']))
        super()._load_agent(options, init_poses)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        scene_name = self.cfg['scene']['name']
        scene_builder = getattr(utils.scenes, f'{scene_name}SceneBuilder')
        self.scene_builder = scene_builder(env=self, cfg=self.cfg)
        self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)


    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    @property
    def middle_agent(self) -> Panda:
        return self.agent.agents[2]
    
    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        pos_C = self.cubeC.pose.p
        offset =  pos_B - pos_A
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeB_on_cubeA = torch.logical_and(xy_flag, z_flag)
        offset = pos_C - pos_B
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeC_on_cubeB = torch.logical_and(xy_flag, z_flag)
        cubeB_to_goal_dist = torch.linalg.norm(
            self.cubeB.pose.p[:, :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        cubeB_placed = cubeB_to_goal_dist < self.goal_radius
        cubeC_to_goal_dist = torch.linalg.norm(
            self.cubeC.pose.p[:, :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        cubeC_placed = cubeC_to_goal_dist < self.goal_radius
        is_cubeA_grasped = self.left_agent.is_grasping(self.cubeA)
        is_cubeB_grasped = self.right_agent.is_grasping(self.cubeB)
        is_cubeC_grasped = self.left_agent.is_grasping(self.cubeC)
        success = (
            is_cubeC_on_cubeB * is_cubeB_on_cubeA * cubeB_placed * cubeC_placed * (~is_cubeA_grasped) * (~is_cubeB_grasped) * (~is_cubeC_grasped)
        )
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeB_grasped": is_cubeB_grasped,
            "is_cubeC_grasped": is_cubeC_grasped,
            "is_cubeA_on_cubeB": is_cubeB_on_cubeA,
            "is_cubeC_on_cubeA": is_cubeC_on_cubeB,
            "cubeB_placed": cubeB_placed,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        return {}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}
