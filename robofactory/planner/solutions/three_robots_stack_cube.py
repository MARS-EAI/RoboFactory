import numpy as np
import sapien

from robofactory.tasks import ThreeRobotsStackCubeEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: ThreeRobotsStackCubeEnv, seed=102, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        is_multi_agent=True
    )
    env = env.unwrapped
    # Task decomposition
    # Step 1: Three robots pick up the cubes
    grasp_poseA = planner.get_grasp_pose_from_obb(env.cubeA, 0)
    grasp_poseB = planner.get_grasp_pose_from_obb(env.cubeB, 1)
    grasp_poseC = planner.get_grasp_pose_from_obb(env.cubeC, 2)
    planner.move_to_pose_with_screw([grasp_poseA, grasp_poseB, grasp_poseC], move_id=[0, 1, 2])
    planner.close_gripper([0, 1, 2])
    grasp_poseA[2] += 0.5
    grasp_poseB[2] += 0.5
    grasp_poseC[2] += 0.5
    planner.move_to_pose_with_screw([grasp_poseA, grasp_poseB, grasp_poseC], move_id=[0, 1, 2])

    # Step 2: Stack the cubes
    target_poseA = planner.get_grasp_pose_for_stack(grasp_poseA, env.goal_region)
    planner.move_to_pose_with_screw(target_poseA, move_id=0)
    planner.open_gripper([0])
    planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
    target_poseB = planner.get_grasp_pose_for_stack(grasp_poseB, env.cubeA)
    planner.move_to_pose_with_screw(target_poseB, move_id=1)
    planner.open_gripper([1])
    planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
    target_poseC = planner.get_grasp_pose_for_stack(grasp_poseC, env.cubeB)
    planner.move_to_pose_with_screw(target_poseC, move_id=2)
    planner.open_gripper([2])

    # Release all grippers
    res = planner.open_gripper([0, 1, 2])
    planner.close()

    return res
