import sys
sys.path.append('./') 
# For custom policy eval
sys.path.insert(0, './custom_policy') 

import os

from collections import defaultdict
from tasks import *

import yaml
from datetime import datetime
from planner.motionplanner import PandaArmMotionPlanningSolver


import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from utils.wrappers.record import RecordEpisodeMA

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

from deploy_policy import DeployPolicy

@dataclass
class Args:
    config: str = "configs/table/place_food.yaml"
    """Configuration to build scenes, assets and agents."""
    
    ckpt_path: str = 'checkpoints/last.ckpt'
    """Path to the checkpoint file of the policy to be evaluated."""

    record_dir: Optional[str] = './eval_video/{env_id}'
    """Directory to save recordings"""

    max_steps: int = 60
    """Maximum number of steps to run the simulation"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    """Observation mode"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    """Control mode"""

    render_mode: str = "sensors"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 10000
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

def main(args: Args):
    # set up the evaluation settings
    np.set_printoptions(suppress=True, precision=5)
    verbose = 0
    if args.seed is not None:
        np.random.seed(args.seed)
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False

    # set env_id and env_kwargs
    env_id = ""
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        task_name = config['task_name']
        print("Evaluation Task:", env_id)
        env_id = task_name + '-rf'
    env_kwargs = dict(
        config=args.config,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    
    # add log_file
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = f"eval_{env_id}_{timestamp}.txt"

    # deploy policy
    model = DeployPolicy(args.ckpt_path)
    
    # evaluate the policy
    total_num = 0
    total_success = 0
    for now_seed in range(args.seed, args.seed + 100):
        # initialize
        model.reset()
        now_success = 0
        total_num += 1
        record_dir = args.record_dir + '_' + str(timestamp) + '/' + str(now_seed)
        np.random.seed(now_seed)

        # build the environment
        print("Current Evaluation Seed: ", now_seed)
        env: BaseEnv = gym.make(env_id, **env_kwargs)
        record_dir = record_dir.format(env_id=env_id)
        env = RecordEpisodeMA(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=30000000)
        env.reset(seed=now_seed)
        planner = PandaArmMotionPlanningSolver(
            env,
            debug=False,
            vis=verbose,
            base_pose=[agent.robot.pose for agent in env.agent.agents],
            visualize_target_grasp_pose=verbose,
            print_env_info=False,
            is_multi_agent=True
        )
        agent_num = planner.agent_num
        if now_seed is not None and env.action_space is not None:
            env.action_space.seed(now_seed)
        if args.render_mode is not None:
            viewer = env.render()
            if isinstance(viewer, sapien.utils.Viewer):
                viewer.paused = args.pause
            env.render()
        
        # update the model with initial observation
        obs = env.get_obs()
        model.update_obs(obs)
        cnt = 0
        while True:
            cnt = cnt + 1
            if cnt > args.max_steps:
                break
            actions = model.get_action()         # list: [action_1, action_2, ..., action_m]; action: [agent_num * 8]

            # For ManiSkill, action to env should be dictionary of all agents
            action_dict = defaultdict(list)

            for t in range(len(actions)):
                action = actions[t]
                raw_obs = env.get_obs()
                for id in range(agent_num):
                    action_dict[f'panda-{id}'].clear()
                    agent_action = action[id * 8 : (id + 1) * 8]

                    # plann true action for agent control
                    current_qpos = raw_obs['agent'][f'panda-{id}']['qpos'].squeeze(0)[:-2].cpu().numpy()
                    path = np.vstack((current_qpos, agent_action[:-1]))
                    try:
                        # important for speed of eval
                        times, position, right_vel, acc, duration = planner.planner[id].TOPP(path, 0.05, verbose=True)
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        action_now = np.hstack([current_qpos, agent_action[-1]])
                        action_dict[f'panda-{id}'].append(action_now)
                        continue
                    n_step = position.shape[0]
                    gripper_state = agent_action[-1]
                    if n_step == 0:
                        action_now = np.hstack([agent_action[:-1], gripper_state])
                        action_dict[f'panda-{id}'].append(action_now)
                    else:
                        for j in range(n_step):
                            action_now = np.hstack([position[j], gripper_state])
                            action_dict[f'panda-{id}'].append(action_now)
                
                # execute actions for all agents
                max_step = 0
                for id in range(agent_num):
                    max_step = max(max_step, len(action_dict[f'panda-{id}']))
                for t_step in range(max_step):
                    true_action = dict()
                    for id in range(agent_num):
                        now_step = min(t_step, len(action_dict[f'panda-{id}']) - 1)
                        true_action[f'panda-{id}'] = action_dict[f'panda-{id}'][now_step]
                    # action execute
                    observation, reward, terminated, truncated, info = env.step(true_action)
                    env.render_human()
                # update obs
                raw_obs = env.get_obs()
                model.update_obs(raw_obs)

            # check if the task is done
            info = env.get_info()
            if args.render_mode is not None:
                env.render()
            if info['success'] == True:
                total_success += 1
                now_success = 1
                env.close()
                print("=================")
                print("Success!, Total Step=", cnt)
                print("=================")
                print(f"Saving video to {record_dir}")
                break
        
        # already to max steps
        with open(log_file, "a") as f:
            f.write(f"\n[Summary] Success Rate: {total_success} / {total_num}\n")
            f.write(f"Current Seeds: {now_seed}, success: {now_success}\n")
        if now_success == 0:
            print("=================")
            print("Failed!")
            print("=================")
            env.close()
        print(f"Saving video to {record_dir}")

if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
