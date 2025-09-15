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
        base_table = self.data['table']#之前记录的table_base
        base_tag_pose = base_table[:3].reshape(-1)
        base_tag_Yaw = base_table[3:].reshape(-1)
        base_knuckle = self.data['tag_tube']#记录的钢管位置,xyz位置+xyzw四元数
        base_knuckle_pose = base_knuckle[:3].reshape(-1) #只取位置信息变成np数组
        base_Yaw =  base_knuckle[3:].reshape(-1)
        ee_pose_t = self.data['ee_pose']#末端位姿
        ee_rot_t = self.data['ee_rot'] #33旋转矩阵
        action_t = self.data['action'].reshape(-1, 1)#原先是手，现在是夹爪,这里的action是优化过的，只跟hand相关
        
        delta = np.zeros(3,dtype = float)
        y_delta = [0,0,0,0]
        if self.frame == 0:
            while rclpy.ok():#查询table_base和tag_frame之间的变换
                try:
                    tag_tf = self.tf2_buffer.lookup_transform(
                         "fake_table_base",'table_base', rclpy.time.Time(), rclpy.duration.Duration(seconds=1))
                    
                    now = self.node.get_clock().now().to_msg()
                    tf_time = tag_tf.header.stamp
                    dt = (rclpy.time.Time.from_msg(now) - rclpy.time.Time.from_msg(tf_time)).nanoseconds * 1e-9

                    if dt > 0.8:
                        self.node.get_logger().warn(
                            f"TF too old (age={dt:.3f}s), waiting for a fresh one..."
                        )
                        time.sleep(0.1)
                        continue
                    
                    t = tag_tf.transform.translation
                    q = tag_tf.transform.rotation
                    tag_pose = np.array([t.x, t.y, t.z])
                    tag_quat = np.array([q.x, q.y, q.z,q.w])
                    self.tag_pose = tag_pose
                    self.tag_quat = tag_quat
                    self.node.get_logger().info(f"Tag pose: {self.tag_pose}")
                    self.node.get_logger().info(f"Tag_quat: {self.tag_quat}")
                    break
                except tf2_ros.LookupException:
                    self.node.get_logger().warn("Transform fr3_table_base->tag_frame not yet available, retrying...")
                    time.sleep(0.5)
            self.table_delta = -(self.tag_pose - base_tag_pose)
            self.y_tag_delta = self.Rcor.compute_rotation_diff(base_tag_Yaw,self.tag_quat , axis='z')
            while rclpy.ok():
                try:
                    knuckle_tf = self.tf2_buffer.lookup_transform(
                        "fake_tag", 'detected_target', rclpy.time.Time(), rclpy.duration.Duration(seconds=1))
                    
                    now = self.node.get_clock().now().to_msg()
                    tf_time = knuckle_tf.header.stamp
                    dt = (rclpy.time.Time.from_msg(now) - rclpy.time.Time.from_msg(tf_time)).nanoseconds * 1e-9

                    if dt > 0.8:
                        self.node.get_logger().warn(
                            f"TF too old (age={dt:.3f}s), waiting for a fresh one..."
                        )
                        time.sleep(0.1)
                        continue
                    
                    t = knuckle_tf.transform.translation
                    q = knuckle_tf.transform.rotation
                    knuckle_pose = np.array([t.x, t.y, t.z])
                    knuckle_quat = np.array([q.x, q.y, q.z,q.w])
                    self.knuckle_pose = knuckle_pose
                    self.knuckle_quat = knuckle_quat
                    self.read_tag = np.array([t.x, t.y, t.z, q.x, q.y, q.z, q.w])
                    self.node.get_logger().info(
                        f"Knuckle pose: {self.knuckle_pose}")
                    break
                except tf2_ros.LookupException:
                    self.node.get_logger().warn(
                        "Transform fake_tag->detected_target not yet available, retrying...")
                    time.sleep(0.5)
            delta = self.knuckle_pose - base_knuckle_pose #现在读到的跟之前记录的delta,只用base_knuckle_pose前三个
            y_delta = self.Rcor.compute_rotation_diff(base_Yaw,self.knuckle_quat,axis='z')#计算绕z轴的旋转差值
            pi_over_2 = np.pi/2 
            if y_delta > pi_over_2 :
                y_delta -= np.pi
                self.step2_turn = 1
            elif y_delta < -pi_over_2:
                y_delta += np.pi
                self.step2_turn = 1
            else:
                self.step2_turn = 0
            print(f"KNUCKLE {delta} {self.knuckle_pose} {base_knuckle_pose}{y_delta}")
            # delta[0], delta[1],delta[2] =  -delta[0],delta[1],-delta[2] #z轴相反，是 -delta[2]
        elif self.frame == 3:
            self.num_steel +=1 
            if self.num_steel >= 3:
                delta[1] = -0.04 * (self.num_steel-3)
                delta[2] = 0.01 +0.025
            else :
                delta[1] = -0.04 * self.num_steel
                delta[2] = 0.01
        else:
            print(f"KNUCKLE {delta} {self.knuckle_pose} {base_knuckle_pose}")
        
        if self.frame == 0:
            # delta[2] = 0.005#0.005 * np.random.uniform(0.8, 1.2)
            # delta[1] = 0.00
            ee_rot_t = self.Rcor.apply_yaw_correction(ee_rot_t, y_delta-self.y_tag_delta)
            # delta[2] -= 0.003
        if self.frame == 1:
            delta[2] = 0.008#0.005 * np.random.uniform(0.8, 1.0)
            # delta[0] -= 0.02
            if self.step2_turn == 1:
                ee_rot_t = self.Rcor.apply_yaw_correction(ee_rot_t, -np.pi-self.y_tag_delta)
            
            ee_rot_t = self.Rcor.apply_pitch_correction(ee_rot_t, -15 ,degrees=True)
            target_rot = self.Rcor.apply_pitch_correction(target_rot, -15 ,degrees=True)

        if self.frame == 2:
            # delta[2] += 0.005#0.005 * np.random.uniform(0.8, 1.0)
            # delta[0] -= 0.025
            if self.step2_turn == 1:
                ee_rot_t = self.Rcor.apply_yaw_correction(ee_rot_t, -np.pi-self.y_tag_delta)
                
        if self.frame == 3:
            if self.step2_turn == 1:
                ee_rot_t = self.Rcor.apply_yaw_correction(ee_rot_t, np.pi-self.y_tag_delta)
        print("Delta", delta)
        delta += self.table_delta
        ee_pose_t += delta[None, :]
        ee_pose_t = self.Rcor.apply_translation_correction(ee_pose_t, self.y_tag_delta)#加上tag_base旋转的平移修正
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