import numpy as np
import sapien
import trimesh

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import to_sapien_pose
from robofactory.utils.mplib_utils import FlexiblePlanner
from .utils import get_actor_obb, compute_grasp_info_by_obb, build_panda_gripper_grasp_pose_visual

from transforms3d import quaternions
from transforms3d.euler import quat2euler, euler2quat
from copy import deepcopy
import transforms3d as t3d
from typing import Union, List, Any
import sapien.physx as physx
OPEN = 1
CLOSED = -1


class PandaArmMotionPlanningSolver:
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: Union[sapien.Pose, List[sapien.Pose]] = None,
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        is_multi_agent: bool = False,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.is_multi_agent = is_multi_agent
        
        if self.is_multi_agent:
            assert hasattr(self.base_env.agent, "agents"), "Multi-agent environment must have agents attribute"
            self.env_agent: List[BaseAgent] = self.base_env.agent.agents
        else:
            self.env_agent: List[BaseAgent] = [self.base_env.agent, ]

        self.agent_num = len(self.env_agent) if is_multi_agent else 1

        self.robot = [agent.robot for agent in self.env_agent]
        self.base_pose = [base_pose, ] if not isinstance(base_pose, list) else base_pose  
        # Assume all agents have the same control mode and joint limits
        self.control_mode = self.env_agent[0].control_mode
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        print("PandaArm Control Mode:", self.control_mode)
        self.planner = self.setup_planner()

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.gripper_state = [OPEN, ] * self.agent_num
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            self.grasp_pose_visual = []
            for id in range(self.agent_num):
                self.grasp_pose_visual.append(build_panda_gripper_grasp_pose_visual(self.base_env.scene, "grasp_pose_visual" + str(id)))
                self.grasp_pose_visual[id].set_pose(self.base_pose[id])
        
        self.elapsed_steps = 0
        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def setup_planner(self):
        planner_group = []
        for id in range(self.agent_num):
            link_names = [link.get_name() for link in self.robot[id].get_links()]
            joint_names = [joint.get_name() for joint in self.robot[id].get_active_joints()]
            planner = FlexiblePlanner(
                urdf=self.env_agent[id].urdf_path,
                srdf=self.env_agent[id].urdf_path.replace(".urdf", ".srdf"),
                user_link_names=link_names,
                user_joint_names=joint_names,
                move_group="panda_hand_tcp",
                joint_vel_limits=np.ones(7) * self.joint_vel_limits,
                joint_acc_limits=np.ones(7) * self.joint_acc_limits,
            )
            self.base_pose[id] = to_sapien_pose(self.base_pose[id])
            planner.set_base_pose(np.hstack([self.base_pose[id].p, self.base_pose[id].q]))
            planner_group.append(planner)
        return planner_group
                    
    def follow_path(self, result_group, move_id, refine_steps: int = 0):
        n_step = 0
        for i in range(len(result_group)):
            n_step = max(n_step, result_group[i]["position"].shape[0])
        # Multi-Agent check collision
        # if check_collision(self, result_group, move_id, n_step, jump):
        #     print("Collision detected")
        #     return False
        for i in range(n_step + refine_steps):
            if not self.is_multi_agent:
                qpos = result_group[0]["position"][min(i, n_step - 1)]
                if self.control_mode == "pd_joint_pos_vel":
                    qvel = result_group[0]["velocity"][min(i, n_step - 1)]
                    action = np.hstack([qpos, qvel, self.gripper_state])
                else:
                    action = np.hstack([qpos, self.gripper_state])
                obs, reward, terminated, truncated, info = self.env.step(action)
            else:
                action_dict = dict()
                for id in range(self.agent_num):
                    if id not in move_id:
                        qpos = self.robot[id].get_qpos()[0, :-2].cpu().numpy()
                        if self.control_mode == "pd_joint_pos":
                            action = np.hstack([qpos, self.gripper_state[id]])
                        else:
                            action = np.hstack([qpos, qpos * 0, self.gripper_state[id]])
                        action_dict[f"panda-{id}"] = action
                    else:
                        move_idx = move_id.index(id)
                        self_step = result_group[move_idx]["position"].shape[0]
                        qpos = result_group[move_idx]["position"][min(i, self_step - 1)]
                        if self.control_mode == "pd_joint_pos_vel":
                            qvel = result_group[move_idx]["velocity"][min(i, self_step - 1)]
                            action = np.hstack([qpos, qvel, self.gripper_state[id]])
                        else:
                            action = np.hstack([qpos, self.gripper_state[id]])
                        action_dict[f"panda-{id}"] = action
                obs, reward, terminated, truncated, info = self.env.step(action_dict)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return True, obs, reward, terminated, truncated, info

    def move_to_pose_with_screw(
        self, pose: Union[Any, List[Any]], dry_run: bool = False, refine_steps: int = 0, move_id : Union[int, List[int]] = 0, jump: int = 1
    ):
        pose = [pose, ] if not isinstance(pose, list) else pose
        pose = [to_sapien_pose(p) for p in pose]
        move_id = [move_id, ] if not isinstance(move_id, list) else move_id
        # try screw two times before giving up
        if self.grasp_pose_visual is not None:
            for id in range(len(pose)):
                self.grasp_pose_visual[move_id[id]].set_pose(pose[id])
            if self.vis:
                self.base_env.render_human()
        result_group = []
        for id in range(len(pose)):
            planner_id = move_id[id]
            # print("planner_id", planner_id, "pose:", pose[id])
            result = self.planner[planner_id].plan_screw(
                np.concatenate([pose[id].p, pose[id].q]),
                self.robot[planner_id].get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            # self.robot[planner_id].set_qpos([-0.0496536, 1.11766171, 0.02375544, -1.74971247, -0.00812339, 4.43305492, 0.8032164, 0.04, 0.04])
            # while True:
            #    self.base_env.render_human()
            if result["status"] != "Success":
                result = self.planner[planner_id].plan_screw(
                    np.concatenate([pose[id].p, pose[id].q]),
                    self.robot[planner_id].get_qpos().cpu().numpy()[0],
                    time_step=self.base_env.control_timestep,
                    use_point_cloud=self.use_point_cloud,
                )
                if result["status"] != "Success":
                    print("Failed to plan screw motion in agent ", id)
                    self.render_wait()
                    return -1
            self.render_wait()
            if dry_run:
                return result
            result_group.append(result)
        return self.follow_path(result_group, move_id, refine_steps=refine_steps)
    
    def open_gripper(self, open_id: Union[int, List[int]] = 0):
        open_id = [open_id, ] if not isinstance(open_id, list) else open_id
        for i in range(20):
            if not self.is_multi_agent:
                self.gripper_state[0] = min(self.gripper_state[0] + 0.1, OPEN)
                qpos = self.robot[0].get_qpos()[0, :-2].cpu().numpy()
                if self.control_mode == "pd_joint_pos":
                    action = np.hstack([qpos, self.gripper_state[0]])
                else:
                    action = np.hstack([qpos, qpos * 0, self.gripper_state[0]])
                obs, reward, terminated, truncated, info = self.env.step(action)
            else:
                action_dict = dict()
                for id in range(self.agent_num):
                    qpos = self.robot[id].get_qpos()[0, :-2].cpu().numpy()
                    if self.control_mode == "pd_joint_pos":
                        action = np.hstack([qpos, self.gripper_state[id]])
                    else:
                        action = np.hstack([qpos, qpos * 0, self.gripper_state[id]])
                    if id in open_id:
                        self.gripper_state[id] = min(self.gripper_state[id] + 0.1, OPEN)
                        action[-1] = self.gripper_state[id]
                    action_dict[f"panda-{id}"] = action
                obs, reward, terminated, truncated, info = self.env.step(action_dict)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, close_id: Union[int, List[int]] = 0):
        close_id = [close_id, ] if not isinstance(close_id, list) else close_id
        # use step-by-step close
        for i in range(20):
            if not self.is_multi_agent:
                self.gripper_state[0] = max(self.gripper_state[0] - 0.1, CLOSED)
                qpos = self.robot[0].get_qpos()[0, :-2].cpu().numpy()
                if self.control_mode == "pd_joint_pos":
                    action = np.hstack([qpos, self.gripper_state[0]])
                else:
                    action = np.hstack([qpos, qpos * 0, self.gripper_state[0]])
                obs, reward, terminated, truncated, info = self.env.step(action)
            else:
                action_dict = dict()
                for id in range(self.agent_num):
                    qpos = self.robot[id].get_qpos()[0, :-2].cpu().numpy()
                    if self.control_mode == "pd_joint_pos":
                        action = np.hstack([qpos, self.gripper_state[id]])
                    else:
                        action = np.hstack([qpos, qpos * 0, self.gripper_state[id]])
                    if id in close_id:
                        self.gripper_state[id] = max(self.gripper_state[id] - 0.1, CLOSED)
                        action[-1] = self.gripper_state[id]
                    action_dict[f"panda-{id}"] = action
                obs, reward, terminated, truncated, info = self.env.step(action_dict)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    """  
        The following method is revised from RoboTwin.  
        It computes the grasp pose by pre-marking the contact points,  
        functional points, etc. and their directions on the object.  
    """  
    def get_target_pose_w_labeled_direction(self, actor, actor_data, pre_dis = 0.0, id = 0):
        actor_matrix = actor.pose.to_transformation_matrix()
        actor_matrix = actor_matrix[0]
        local_target_matrix = np.asarray(actor_data['target_pose'][id])
        local_target_matrix[:3,3] *= actor_data['scale']
        convert_matrix = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
        global_target_pose_matrix = actor_matrix.cpu().numpy() @ local_target_matrix @ convert_matrix
        global_target_pose_matrix_q = global_target_pose_matrix[:3,:3]
        global_target_pose_p = global_target_pose_matrix[:3,3] + global_target_pose_matrix_q @ np.array([pre_dis,0,0]).T
        global_target_pose_q = t3d.quaternions.mat2quat(global_target_pose_matrix_q)
        res_pose = list(global_target_pose_p)+list(global_target_pose_q)
        return np.array(res_pose)
        
    def get_grasp_pose_w_labeled_direction(self, actor, actor_data, pre_dis = 0.0, id = 0):
        actor_matrix = actor.pose.to_transformation_matrix()
        actor_matrix = actor_matrix[0]
        local_contact_matrix = np.asarray(actor_data['contact_points_pose'][id])
        local_contact_matrix[:3,3] *= actor_data['scale']
        convert_matrix = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
        global_contact_pose_matrix = actor_matrix.cpu().numpy() @ local_contact_matrix @ convert_matrix
        global_contact_pose_matrix_q = global_contact_pose_matrix[:3,:3]
        global_grasp_pose_p = global_contact_pose_matrix[:3,3] + global_contact_pose_matrix_q @ np.array([pre_dis,0,0]).T
        global_grasp_pose_q = t3d.quaternions.mat2quat(global_contact_pose_matrix_q)
        res_pose = list(global_grasp_pose_p)+list(global_grasp_pose_q)
        return np.array(res_pose)
    
    def is_plan_success(self, grasp_pose: list, robot_id : int = 0):
        las_robot_qpose = self.planner[robot_id].robot.get_qpos()
        result = self.planner[robot_id].plan_screw(
            target_pose=grasp_pose,
            qpos=self.robot[robot_id].get_qpos().cpu().numpy()[0],
            time_step=1 / 250,
            use_point_cloud=False,
            use_attach=False,
        )
        self.planner[robot_id].robot.set_qpos(las_robot_qpose,full=True)
        return result["status"] == "Success" and result["position"].shape[0] <= 2000

    def evaluate_grasp_pose(self, endpose, grasp_pose: list, agent_id : int = 0):
        is_plan_suc = self.is_plan_success(grasp_pose=grasp_pose, robot_id=agent_id)
        if not is_plan_suc:
            return -1e10
    
        res = 0
        res += np.sqrt(np.sum((np.array(endpose.p[0]) - np.array(grasp_pose)[:3]) ** 2)) / 0.7
        trans_now_pose_matrix = t3d.quaternions.quat2mat(grasp_pose[3:]) @ np.linalg.inv(endpose.to_transformation_matrix()[0][:3,:3])
        theta_xy = np.mod(np.abs(t3d.euler.mat2euler(trans_now_pose_matrix)[:2]), np.pi)
        res += 2 * np.sum(theta_xy/np.pi)
        return -res
      
    def get_grasp_pose_from_goal_point_and_direction(self, actor, actor_data, endpose, actor_functional_point_id = 0, target_point = None,
                                                     target_approach_direction = [0,0,1,0], pre_dis = 0.):
        target_approach_direction_mat = t3d.quaternions.quat2mat(target_approach_direction)
        actor_matrix = actor.pose.to_transformation_matrix()[0]
        if type(target_point) == np.ndarray:
            target_point_copy = target_point
        else:
            target_point_copy = deepcopy(target_point.raw_pose[0][:3])
        target_point_copy -= target_approach_direction_mat @ np.array([0,0,pre_dis])
        adjunction_matrix_list = [
            # 90 degree
            t3d.euler.euler2mat(0,0,0),
            t3d.euler.euler2mat(0,0,np.pi/2),
            t3d.euler.euler2mat(0,0,-np.pi/2),
            t3d.euler.euler2mat(0,0,np.pi),
            # 45 degree
            t3d.euler.euler2mat(0,0,np.pi/4),
            t3d.euler.euler2mat(0,0,np.pi*3/4),
            t3d.euler.euler2mat(0,0,-np.pi*3/4),
            t3d.euler.euler2mat(0,0,-np.pi/4),
        ]

        res_pose = None
        res_eval= -1e10
        for adjunction_matrix in adjunction_matrix_list:
            local_target_matrix = np.asarray(actor_data['functional_matrix'][actor_functional_point_id])
            local_target_matrix[:3,3] *= actor_data['scale']
            fuctional_matrix = actor_matrix[:3,:3] @ np.asarray(actor_data['functional_matrix'][actor_functional_point_id])[:3,:3]
            fuctional_matrix = fuctional_matrix @ adjunction_matrix
            trans_matrix = target_approach_direction_mat @ np.linalg.inv(fuctional_matrix)
            end_effector_pose_matrix = t3d.quaternions.quat2mat(endpose.q[0]) @ np.array([[1,0,0],[0,1,0],[0,0,1]])
            target_grasp_matrix = trans_matrix @ end_effector_pose_matrix
         
            res_matrix = np.eye(4)
            res_matrix[:3,3] = (actor_matrix  @ local_target_matrix)[:3,3] - endpose.p[0]
            res_matrix[:3,3] = np.linalg.inv(end_effector_pose_matrix) @ res_matrix[:3,3]
            target_grasp_qpose = t3d.quaternions.mat2quat(target_grasp_matrix)
            now_pose = (target_point_copy - target_grasp_matrix @ res_matrix[:3,3]).tolist() + target_grasp_qpose.tolist()
            now_pose_eval = self.evaluate_grasp_pose(endpose, now_pose)
            if not self.is_plan_success(now_pose):
                continue
            elif now_pose_eval > res_eval:
                res_pose = now_pose
                res_eval = now_pose_eval
            else:
                res_pose = now_pose
        if res_pose is None:
            return None
        return np.array(res_pose)

    """  
        The following method is revised from the official ManiSkill code. 
        It constructs an OBB (Oriented Bounding Box) for the object  
        and computes the grasp pose based on approaching, closing, and center.  
    """ 
    def get_grasp_pose_from_obb(self, actor, agent_id : int = 0):
        obb = get_actor_obb(actor)
        approaching = np.array([0, 0, -1])
        target_closing = self.env_agent[agent_id].tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
        FINGER_LENGTH = 0.02 + 0.005
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = self.env_agent[agent_id].build_grasp_pose(approaching, closing, center)
        # search for a valid grasp pose
        angles = np.arange(0, 2 * np.pi / 3, np.pi / 2)
        now_res = -1e11
        endpose = self.env_agent[agent_id].tcp.pose
        for angle in angles:
            delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
            grasp_pose2 = grasp_pose * delta_pose
            grasp_list = list(grasp_pose2.p) + list(grasp_pose2.q)
            is_plan_suc = self.is_plan_success(grasp_pose=grasp_list, robot_id=agent_id)
            if not is_plan_suc:
                continue
            else:
                grasp_pose = grasp_pose2
                now_res = 0
                break
        if now_res < -1e10:
            raise RuntimeError(f"Failed to find a valid grasp pose for {actor.name}")
        return np.array(list(grasp_pose.p) + list(grasp_pose.q))
    
    def get_grasp_pose_for_stack(self, now_pose, target_actor, height_offset=0.05):
        target_pose = target_actor.pose
        target_pose = target_pose * sapien.Pose([0, 0, height_offset])
        offset = (target_pose.p - now_pose[:3]).numpy()[0]
        align_pose = sapien.Pose(now_pose[:3] + offset, now_pose[3:])
        return np.array(list(align_pose.p) + list(align_pose.q))

    def close(self):
        pass
