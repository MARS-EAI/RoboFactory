import numpy as np
import sapien

from robofactory.tasks import PlaceFoodEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: PlaceFoodEnv, seed=None, debug=False, vis=False):
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
    pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.pot, actor_data=env.annotation_data['pot'], pre_dis=0, id=3)
    pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.meat, actor_data=env.annotation_data['meat'], pre_dis=0, id=0)
    pose1[2] += 0.2
    pose2[2] += 0.15
    planner.move_to_pose_with_screw([pose2, pose1], move_id=[0, 1])
    pose1[2] -= 0.22
    pose2[2] -= 0.15
    planner.move_to_pose_with_screw([pose2, pose1], move_id=[0, 1])
    planner.close_gripper(close_id=[0, 1])
    pose1[2] += 0.3
    pose2[2] += 0.15
    planner.move_to_pose_with_screw([pose2, pose1], move_id=[0, 1])
    pose3 = planner.get_target_pose_w_labeled_direction(actor=env.pot, actor_data=env.annotation_data['pot'], pre_dis=0)
    planner.move_to_pose_with_screw(pose3, move_id=[0])
    planner.open_gripper(open_id=[0])
    planner.open_gripper(open_id=[0])
    res = planner.open_gripper(open_id=[0])
    planner.close()
    return res
