from rclpy.node import Node
import tf2_ros
from glob import glob
import numpy as np
import time
import rclpy
from openarm_remote.utils.traj import TrajectoryGenerator, calculate_relative_actions
from openarm_remote.robot_control.mod_ik import General_ArmIK
from openarm_remote.utils.quat_delta import BaseRotationCorrector
import yaml
from ament_index_python.packages import get_package_share_directory
import os
# REPLAY_FILES = sorted(glob('/home/usyd/openarm_ws/detect_record/simple_*'))
MID_Q = np.array([0.11010828 , 0.11387246 ,-0.22371638 ,-2.10449131 ,-0.02231137 , 2.14695302,0.65245617])
MID2_Q = np.array([0.0386087, -0.0869798 ,-0.188516184 ,-1.69528422 ,-0.03064684 , 1.66447923,0.65245617]) #-0.8544360
#只有一段时frame_q两个值要相等
FRAME_Q = [
    [0.000725484974137, 0.0009276392014956, 0.026339594595690147, -0.0543363355379666, -0.10479729101776056, 0.00577383304603, -0.004570819420221],
]

class DetectPolicy:
    def __init__(self, node: Node, ik:General_ArmIK ,arm_id :str):
        self.node = node
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer, node)
        self.ik = ik
        self.Rcor = BaseRotationCorrector(decimal_precision=6) #修正旋转矩阵的类
        self.step_id = 0
        self.actions = []
        self.read_tag = None
        self.num_steel = -1
        self.step2_turn = 0
        self.trajectory_generator = TrajectoryGenerator(hand_wait_time=0.8, keyframe_wait_time=0.0, eps_min=1e-3, eps_max=1e-3, acceleration=5e-6)
        self.fast_trajectory_generator = TrajectoryGenerator(hand_wait_time=0.8, keyframe_wait_time=0.0,eps_min=2e-3, eps_max=3e-3, acceleration=5e-6)
        self.slow_trajectory_generator = TrajectoryGenerator(hand_wait_time=0.8, keyframe_wait_time=0.0, eps_min=1e-3, eps_max=1e-3, acceleration=5e-6)
        self.arm_id = arm_id 
        package_share_directory = get_package_share_directory('openarm_remote')
        config_path = os.path.join(package_share_directory, 'config', 'record_path.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        ws_path = config['files_path']['ws_path']
        base_path = os.path.join(ws_path, "detect_record")
        if arm_id == "left":
            pattern = f"{base_path}/left_arm/simple_*"
        elif arm_id == "right":
            pattern = f"{base_path}/right_arm/simple_*"
        else:
            raise ValueError(f"Unknown arm_id: {arm_id}")
        self.replay_files = sorted(glob(pattern))
    def step(self, obs):
        if self.step_id >= len(self.actions):
            return None,self.read_tag
        
        action = self.actions[self.step_id]
        self.step_id += 1
        return action ,self.read_tag
    
    def reset(self, frame, obs): ##这个可以通过frame跟reset_noMIDQ合并在一起，这里先不合并方便区分
        self.data = np.load(self.replay_files[frame])
        self.step_id = 0
        action_t = self.data['action']
        arm_id = self.data['arm_id'][0]
        if self.arm_id != arm_id :
            self.node.get_logger().warn(f"Policy for '{self.arm_id}' is loading a file for '{arm_id}'.")
        self.frame = frame
        
        return {
            'q': MID_Q,
            'arm_id': self.arm_id
        }
    def reset_noMIDQ(self, frame, obs):
        self.data = np.load(self.replay_files[frame])
        self.step_id = 0
        action_t = self.data['action']
        arm_id = self.data['arm_id'][0]
        if self.arm_id != arm_id :
            self.node.get_logger().warn(f"Policy for '{self.arm_id}' is loading a file for '{arm_id}'.")
        self.frame = frame
        
        return {
            'q': None,
            'arm_id': self.arm_id
        }
    
    
    def post_reset(self, obs):
        target_pose, target_rot = obs['action.ee_pose'], obs['action.ee_rot'].reshape(3, 3)
        # base_knuckle = self.data['tag_tube']#记录的钢管位置,xyz位置+xyzw四元数
        # base_knuckle_pose = base_knuckle[:3].reshape(-1) #只取位置信息变成np数组
        # base_Yaw =  base_knuckle[3:].reshape(-1)
        ee_pose_t = self.data['ee_pose']#末端位姿
        ee_rot_t = self.data['ee_rot'] #33旋转矩阵
        action_t = self.data['action'].reshape(-1, 1)#原先是手，现在是夹爪,这里的action是优化过的，只跟hand相关
        
        delta = np.zeros(3,dtype = float)
        y_delta = [0,0,0,0]
        print("Delta", delta)
        ee_pose_t += delta[None, :]
        ee_rot_t = ee_rot_t.reshape(-1, 3, 3)
        
        if self.frame == 0:
            ee_pose_t[0][0] -= 0.0
        
            mid_fast_ee_pose_t = np.vstack([target_pose.reshape(1, 3),  ee_pose_t[0].reshape(1, 3)])#([target_pose.reshape(1, 3), ee_mid_pose.reshape(1, 3), ee_pose_t[0].reshape(1, 3)])
            mid_fast_ee_rot_t = np.vstack([target_rot.reshape(1, 3, 3),  ee_rot_t[0].reshape(1, 3, 3)])#([target_rot.reshape(1, 3, 3), ee_mid_rot.reshape(1, 3, 3), ee_rot_t[0].reshape(1, 3, 3)])
            mid_fast_action_t = np.vstack([action_t[0], action_t[0]])#([action_t[0], action_t[0], action_t[0]])
            #快速回到初始点
            mid_fast_new_pose, mid_fast_new_rot, mid_fast_new_action = self.fast_trajectory_generator.generate(
                mid_fast_ee_pose_t, mid_fast_ee_rot_t.reshape(-1, 9), mid_fast_action_t
            )
            #计算
            mid_fast_new_rot = mid_fast_new_rot.reshape(-1, 3, 3)
            mid_fast_actions = calculate_relative_actions(
                mid_fast_new_pose, mid_fast_new_rot, mid_fast_new_action, euler_convention='xyz'
            )
        elif self.frame == 1:
            mid_fast_ee_pose_t = np.vstack([target_pose.reshape(1, 3),  ee_pose_t[0].reshape(1, 3)])#([target_pose.reshape(1, 3), ee_mid_pose.reshape(1, 3), ee_pose_t[0].reshape(1, 3)])
            mid_fast_ee_rot_t = np.vstack([target_rot.reshape(1, 3, 3),  ee_rot_t[0].reshape(1, 3, 3)])#([target_rot.reshape(1, 3, 3), ee_mid_rot.reshape(1, 3, 3), ee_rot_t[0].reshape(1, 3, 3)])
            mid_fast_action_t = np.vstack([action_t[0], action_t[0]])#([action_t[0], action_t[0], action_t[0]])
            #快速回到初始点
            mid_fast_new_pose, mid_fast_new_rot, mid_fast_new_action = self.slow_trajectory_generator.generate(
                mid_fast_ee_pose_t, mid_fast_ee_rot_t.reshape(-1, 9), mid_fast_action_t
            )
            #计算
            mid_fast_new_rot = mid_fast_new_rot.reshape(-1, 3, 3)
            mid_fast_actions = calculate_relative_actions(
                mid_fast_new_pose, mid_fast_new_rot, mid_fast_new_action, euler_convention='xyz'
            )
        else:
            mid_fast_ee_pose_t = np.vstack([target_pose.reshape(1, 3),  ee_pose_t[0].reshape(1, 3)])
            mid_fast_ee_rot_t = np.vstack([target_rot.reshape(1, 3, 3),  ee_rot_t[0].reshape(1, 3, 3)])
            mid_fast_action_t = np.vstack([action_t[0],  action_t[0]])
            #第二次夹起钢管慢点渠道第一个点
            mid_fast_new_pose, mid_fast_new_rot, mid_fast_new_action = self.fast_trajectory_generator.generate(
                mid_fast_ee_pose_t, mid_fast_ee_rot_t.reshape(-1, 9), mid_fast_action_t
            )
            #计算
            mid_fast_new_rot = mid_fast_new_rot.reshape(-1, 3, 3)
            mid_fast_actions = calculate_relative_actions(
                mid_fast_new_pose, mid_fast_new_rot, mid_fast_new_action, euler_convention='xyz'
            )
        new_pose, new_rot, new_action = self.trajectory_generator.generate_Z5cm(
            ee_pose_t, ee_rot_t.reshape(-1, 9), action_t
        )
        new_rot = new_rot.reshape(-1, 3, 3)
        actions = calculate_relative_actions(
            new_pose, new_rot, new_action, euler_convention='xyz'
        )

        self.actions = np.vstack([mid_fast_actions, actions ])
        return self.actions
        # self.actions[:, :3] /= 0.005
        # self.actions[:, 3:6] /= 0.01