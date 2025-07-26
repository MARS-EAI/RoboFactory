from typing import Any, Dict, Tuple
import os.path as osp
import numpy as np
import sapien
import torch
import json
import yaml

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
import robofactory.utils.scenes as scene_rf
from robofactory import CONFIG_DIR
from robofactory.utils.nested_dict_utils import nested_yaml_map, replace_dir

@register_env("TakePhoto-rf", max_episode_steps=1500)
class TakePhotoEnv(BaseEnv):

    SUPPORTED_ROBOTS = [("panda", "panda", "panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda, Panda, Panda]]
    goal_thresh = 0.025

    def __init__(
        self, *args, robot_uids=("panda", "panda", "panda"), robot_init_qpos_noise=0.02, **kwargs
    ):
        if 'config' in kwargs:
            with open(kwargs['config'], 'r', encoding='utf-8') as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            del kwargs['config']
        else:
            if 'scene' in kwargs:
                scene = kwargs['scene']
                del kwargs['scene']
            else:
                scene = 'table'
            with open(osp.join(CONFIG_DIR, scene, 'take_photo.yaml'), 'r', encoding='utf-8') as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.cfg = nested_yaml_map(replace_dir, cfg)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

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
        scene_name = self.cfg['scene']['name']
        scene_builder = getattr(scene_rf, f'{scene_name}SceneBuilder')
        self.scene_builder = scene_builder(env=self, cfg=self.cfg)
        self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            # alignment between cube and camera
            cube_ppose = self.cube.pose.p
            temp_pose = self.camera.pose
            temp_pose.p[:, 0] = cube_ppose[:, 0]
            temp_pose.p[:, 1] = cube_ppose[:, 1]
            self.camera.set_pose(temp_pose)

    def evaluate(self):
        camera = getattr(self, 'camera')
        meat = getattr(self, 'meat')
        camera_button_pose = camera.pose.p[..., :2]
        camera_button_pose[..., 0] += 0.035
        camera_button_pose[..., 1] -= 0.09
        camera_to_agent_pose = self.agent.agents[3].tcp.pose.p[..., :2] - camera_button_pose
        camera_hanging = camera.pose.p[..., 2] > self.agent.agents[0].robot.pose.p[0, 2] + 0.20
        meat_hanging = meat.pose.p[..., 2] > self.agent.agents[0].robot.pose.p[0, 2] + 0.20
        meat_to_agent_pose = self.agent.agents[2].tcp.pose.p[..., :2] - meat.pose.p[..., :2]
        print("camera_to_agent_pose: ", camera_to_agent_pose)
        print("meat_to_agent_pose: ", meat_to_agent_pose)
        success = torch.all(torch.abs(camera_to_agent_pose) < 0.035, dim=1) and camera_hanging and meat_hanging and meat_to_agent_pose[..., 1] < 0.08
        return {
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        return {}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return []
    
