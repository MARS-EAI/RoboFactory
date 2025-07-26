import numpy as np
import sapien

from robofactory.tasks import StackCubeEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: StackCubeEnv, seed=None, debug=False, vis=False):
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
    pose1 = planner.get_grasp_pose_from_obb(env.cubeA)
    pose1[2] -= 0.02
    planner.move_to_pose_with_screw(pose1)
    pose1[2] += 0.02
    planner.move_to_pose_with_screw(pose1)
    planner.close_gripper()
    pose2 = planner.get_grasp_pose_for_stack(pose1, env.cubeB)
    pose2[2] += 0.02
    planner.move_to_pose_with_screw(pose2)
    res = planner.open_gripper()
    planner.close()
    return res
