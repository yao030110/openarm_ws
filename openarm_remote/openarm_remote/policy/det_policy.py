from rclpy.node import Node
import tf2_ros
from glob import glob
import numpy as np
import time
import rclpy
from openarm_remote.utils.traj import TrajectoryGenerator, calculate_relative_actions
from openarm_remote.robot_control.mod_ik import General_ArmIK
from openarm_remote.utils.quat_delta import BaseRotationCorrector
REPLAY_FILES = sorted(glob('/home/usyd/openarm_ws/detect_record/simple_*'))
MID_Q = np.array([0.11010828 , 0.11387246 ,-0.22371638 ,-2.10449131 ,-0.02231137 , 2.14695302,0.65245617])
MID2_Q = np.array([0.0386087, -0.0869798 ,-0.188516184 ,-1.69528422 ,-0.03064684 , 1.66447923,0.65245617]) #-0.8544360
#只有一段时frame_q两个值要相等
FRAME_Q = [
    [0.380725484974137, 0.2709276392014956, 0.026339594595690147, -1.7543363355379666, -0.10479729101776056, 1.928577383304603, -0.2834570819420221],
    [-0.025618523423475717, 0.18571219564933367, -0.22699352724584368, -1.9323872277499108, -0.10714389966252504, 2.0706494030655054, -0.9237536021706692]
]

class DetectPolicy:
    def __init__(self, node: Node, ik:General_ArmIK):
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
        self.trajectory_generator = TrajectoryGenerator(hand_wait_time=0.8, keyframe_wait_time=0.0, eps_min=1e-2, eps_max=2e-2, acceleration=5e-6)
        self.fast_trajectory_generator = TrajectoryGenerator(hand_wait_time=0.8, keyframe_wait_time=0.0, eps_min=2e-2, eps_max=3e-2, acceleration=5e-6)
        self.slow_trajectory_generator = TrajectoryGenerator(hand_wait_time=0.8, keyframe_wait_time=0.0, eps_min=8e-3, eps_max=1e-2, acceleration=5e-6)
    def step(self, obs):
        if self.step_id >= len(self.actions):
            return None,self.read_tag
        
        action = self.actions[self.step_id]
        self.step_id += 1
        return action ,self.read_tag
    
    def reset(self, frame, obs): ##这个可以通过frame跟reset_noMIDQ合并在一起，这里先不合并方便区分
        self.data = np.load(REPLAY_FILES[frame])
        self.step_id = 0
        action_t = self.data['action']
        
        self.frame = frame
        
        return {
            'q': MID_Q,
        }
    def reset_noMIDQ(self, frame, obs):
        self.data = np.load(REPLAY_FILES[frame])
        self.step_id = 0
        
        self.frame = frame
        
        return {
            'q':None,
        }
    
    
    def post_reset(self, obs):
        target_pose, target_rot = obs['action.ee_pose'], obs['action.ee_rot'].reshape(3, 3)#是simp文件和
        ee_pose_t = self.data['ee_pose']#末端位姿
        ee_rot_t = self.data['ee_rot'] #33旋转矩阵
        # action_t = self.data['action'].reshape(-1, 1)#原先是手，现在是夹爪,这里的action是优化过的，只跟hand相关
        action_t = np.array([0.0]).reshape(-1, 1) 
        
        delta = np.zeros(3,dtype = float)
        y_delta = [0,0,0,0]
        
        print("Delta", delta)
        delta += self.table_delta
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

        # self.actions[:, :3] /= 0.005
        # self.actions[:, 3:6] /= 0.01