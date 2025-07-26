import numpy as np
import sapien

from robofactory.tasks import PickMeatEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: PickMeatEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    env = env.unwrapped
    pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.meat, actor_data=env.annotation_data['meat'], pre_dis=0)
    planner.move_to_pose_with_screw(pose1)
    planner.close_gripper()
    pose1[2] += 0.2
    res = planner.move_to_pose_with_screw(pose1)
    planner.close()
    return res
