import numpy as np
import sapien

from robofactory.tasks import LiftBarrierEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: LiftBarrierEnv, seed=None, debug=False, vis=False):
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
    pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.barrier, actor_data=env.annotation_data['barrier'], pre_dis=0, id=1)
    pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.barrier, actor_data=env.annotation_data['barrier'], pre_dis=0, id=2)
    planner.move_to_pose_with_screw(pose=[pose1, pose2], move_id=[0, 1])
    planner.close_gripper(close_id=[0, 1])
    pose1[2] += 0.2
    pose2[2] += 0.2
    planner.move_to_pose_with_screw(pose=[pose1, pose2], move_id=[0, 1])
    res = planner.close_gripper()
    planner.close()
    return res
