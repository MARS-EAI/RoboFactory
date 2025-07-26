import numpy as np
import sapien

from robofactory.tasks import TakePhotoEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: TakePhotoEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=False,
        print_env_info=False,
        is_multi_agent=True
    )
    env=env.unwrapped
    pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.camera, actor_data=env.annotation_data['camera'], pre_dis=0, id=1)
    pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.camera, actor_data=env.annotation_data['camera'], pre_dis=0, id=0)
    pose3 = planner.get_grasp_pose_w_labeled_direction(actor=env.meat, actor_data=env.annotation_data['meat'], pre_dis=0)
    pose3[2] += 0.1
    planner.move_to_pose_with_screw([pose1,pose2,pose3], move_id=[0,1,2])
    pose3[2] -= 0.1
    planner.move_to_pose_with_screw([pose3], move_id=[2])
    planner.close_gripper(close_id=[2])
    pose1[2] += 0.15
    pose2[2] += 0.15
    pose3[0] = env.agent.agents[2].robot.pose.p[0, 0] - 0.7
    pose3[1] = env.agent.agents[2].robot.pose.p[0, 1]
    pose3[2] += 0.25
    res = planner.move_to_pose_with_screw([pose1,pose2,pose3], move_id=[0,1,2])
    pose4 = planner.get_grasp_pose_w_labeled_direction(actor=env.camera, actor_data=env.annotation_data['camera'], pre_dis=0, id=2)
    planner.close_gripper(close_id=[3])
    res = planner.move_to_pose_with_screw([pose4], move_id=[3])
    pose4[2] -= 0.03
    res = planner.move_to_pose_with_screw([pose4], move_id=[3])
    planner.close()
    return res
