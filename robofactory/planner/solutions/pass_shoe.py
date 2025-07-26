import numpy as np
import sapien
import torch

from robofactory.tasks import PassShoeEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: PassShoeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        is_multi_agent=True
    )
    env = env.unwrapped
    pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.shoe, actor_data=env.annotation_data['shoe'], pre_dis=0, id=1)
    pose1[2] += 0.12
    planner.move_to_pose_with_screw(pose1, move_id=[0])
    pose1[2] -= 0.12
    planner.move_to_pose_with_screw(pose1, move_id=[0])
    planner.close_gripper(close_id=[0])
    pose1[2] += 0.05
    planner.move_to_pose_with_screw(pose1, move_id=[0])
    pose1[:2] = (env.agent.agents[0].robot.pose.p[0, :2] + env.agent.agents[1].robot.pose.p[0, :2]) / 2
    planner.move_to_pose_with_screw(pose1, move_id=[0])
    planner.open_gripper(open_id=[0])
    # avoid collision
    pose1[:2] -= 0.15
    pose1[2] = env.agent.agents[0].robot.pose.p[0, 2] + 0.3
    planner.move_to_pose_with_screw(pose1, move_id=[0])
    pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.shoe, actor_data=env.annotation_data['shoe'], pre_dis=0, id=0)
    pose2[2] += 0.12
    planner.move_to_pose_with_screw(pose2, move_id=[1])
    pose2[2] -= 0.12
    planner.move_to_pose_with_screw(pose2, move_id=[1])
    planner.close_gripper(close_id=[1])
    pose2[2] += 0.05
    planner.move_to_pose_with_screw(pose2, move_id=[1])
    pose2[:2] = env.goal_region.pose.p[0, :2]
    planner.move_to_pose_with_screw(pose2, move_id=[1])
    res = planner.open_gripper(open_id=[1])
    planner.close()
    return res
