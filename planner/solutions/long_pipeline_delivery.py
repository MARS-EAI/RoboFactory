import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation as R

from tasks import LongPipelineDeliveryEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def random_position_offset():
    return np.random.normal(0, 0.015, 2)

def random_rotation_offset():
    angle = np.random.normal(0, 4)
    angle = np.clip(angle, -15, 15)
    return R.from_euler('x', angle, degrees=True)

def apply_random_rotation(original_quat):
    original_rot = R.from_quat(original_quat)
    random_rot = random_rotation_offset()
    final_rot = random_rot * original_rot
    return final_rot.as_quat()

def solve(env: LongPipelineDeliveryEnv, seed=None, debug=False, vis=False):
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
    planner.move_to_pose_with_screw(pose1, move_id=[3])
    pose1[2] -= 0.12
    planner.move_to_pose_with_screw(pose1, move_id=[3])
    planner.close_gripper(close_id=[3])
    pose1[2] += 0.06
    planner.move_to_pose_with_screw(pose1, move_id=[3])
    pose1[0] = env.agent.agents[3].robot.pose.p[0, 0] + 0.6 + random_position_offset()[0]
    pose1[1] = (env.agent.agents[3].robot.pose.p[0, 1] + env.agent.agents[2].robot.pose.p[0, 1]) / 2 + random_position_offset()[1]
    pose1[3:] = apply_random_rotation(pose1[3:])
    planner.move_to_pose_with_screw(pose1, move_id=[3])
    planner.open_gripper(open_id=[3])
    pose1[2] = env.agent.agents[3].robot.pose.p[0, 2] + 0.2
    planner.move_to_pose_with_screw(pose1, move_id=[3])
    pose1[2] += 0.4
    pose1[1] += 0.3
    planner.move_to_pose_with_screw(pose1, move_id=[3])

    pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.shoe, actor_data=env.annotation_data['shoe'], pre_dis=0, id=1)
    pose2[2] += 0.12
    planner.move_to_pose_with_screw(pose2, move_id=[2])
    pose2[2] -= 0.12
    planner.move_to_pose_with_screw(pose2, move_id=[2])
    planner.close_gripper(close_id=[2])
    pose2[2] += 0.06
    planner.move_to_pose_with_screw(pose2, move_id=[2])
    pose2[0] = env.agent.agents[2].robot.pose.p[0, 0] + 0.6 + random_position_offset()[0]
    pose2[1] = (env.agent.agents[2].robot.pose.p[0, 1] + env.agent.agents[1].robot.pose.p[0, 1]) / 2 + random_position_offset()[1]
    pose2[3:] = apply_random_rotation(pose2[3:])
    planner.move_to_pose_with_screw(pose2, move_id=[2])
    planner.open_gripper(open_id=[2])
    pose2[2] = env.agent.agents[2].robot.pose.p[0, 2] + 0.2
    planner.move_to_pose_with_screw(pose2, move_id=[2])
    pose2[1] += 0.3
    pose2[2] += 0.4
    planner.move_to_pose_with_screw(pose2, move_id=[2])

    pose3 = planner.get_grasp_pose_w_labeled_direction(actor=env.shoe, actor_data=env.annotation_data['shoe'], pre_dis=0, id=1)
    pose3[2] += 0.12
    planner.move_to_pose_with_screw(pose3, move_id=[1])
    pose3[2] -= 0.12
    planner.move_to_pose_with_screw(pose3, move_id=[1])
    planner.close_gripper(close_id=[1])
    pose3[2] += 0.06
    planner.move_to_pose_with_screw(pose3, move_id=[1])
    pose3[0] = env.agent.agents[1].robot.pose.p[0, 0] + 0.6 + random_position_offset()[0]
    pose3[1] = (env.agent.agents[1].robot.pose.p[0, 1] + env.agent.agents[0].robot.pose.p[0, 1]) / 2 + random_position_offset()[1]
    pose3[3:] = apply_random_rotation(pose3[3:])
    planner.move_to_pose_with_screw(pose3, move_id=[1])
    planner.open_gripper(open_id=[1])
    pose3[2] = env.agent.agents[1].robot.pose.p[0, 2] + 0.2
    planner.move_to_pose_with_screw(pose3, move_id=[1])
    pose3[2] += 0.4
    pose3[1] += 0.3
    planner.move_to_pose_with_screw(pose3, move_id=[1])

    pose4 = planner.get_grasp_pose_w_labeled_direction(actor=env.shoe, actor_data=env.annotation_data['shoe'], pre_dis=0, id=1)
    pose4[2] += 0.12
    planner.move_to_pose_with_screw(pose4, move_id=[0])
    pose4[2] -= 0.12
    planner.move_to_pose_with_screw(pose4, move_id=[0])
    planner.close_gripper(close_id=[0])
    pose4[2] += 0.06
    planner.move_to_pose_with_screw(pose4, move_id=[0])
    pose4[:2] = env.goal_region.pose.p[0, :2] + random_position_offset()
    pose4[3:] = apply_random_rotation(pose4[3:])
    planner.move_to_pose_with_screw(pose4, move_id=[0])
    planner.open_gripper(open_id=[0])
    pose4[2] = env.agent.agents[0].robot.pose.p[0, 2] + 0.2
    res = planner.move_to_pose_with_screw(pose4, move_id=[0])
    planner.close()
    return res
