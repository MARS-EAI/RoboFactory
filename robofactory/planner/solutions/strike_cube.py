import numpy as np
import sapien

from robofactory.tasks import StrikeCubeEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: StrikeCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    LIFT_HEIGHT = 0.2
    PRE_DIS = 0.05
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    env = env.unwrapped
    pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.hammer, actor_data=env.annotation_data['hammer'], pre_dis=-0.05) # hammer pre grap pose
    pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.hammer, actor_data=env.annotation_data['hammer'], pre_dis=0) # hammer grap pose
    planner.move_to_pose_with_screw(pose1)
    planner.move_to_pose_with_screw(pose2)
    planner.close_gripper()
    pose2[2] += LIFT_HEIGHT
    planner.move_to_pose_with_screw(pose2)
    pre_strike_block_pose = planner.get_grasp_pose_from_goal_point_and_direction(
                                actor=env.hammer, 
                                actor_data=env.annotation_data['hammer'], 
                                endpose=env.agent.tcp.pose, 
                                actor_functional_point_id=0, 
                                target_point=env.cube.pose[:3],
                                pre_dis=PRE_DIS
                            )
    if pre_strike_block_pose is not None:
        planner.move_to_pose_with_screw(pre_strike_block_pose)
        pre_strike_block_pose[2] -= PRE_DIS
        planner.move_to_pose_with_screw(pre_strike_block_pose)
    res = planner.close_gripper()
    planner.close()
    return res
