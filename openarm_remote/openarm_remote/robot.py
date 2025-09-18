from rclpy.node import Node
import rclpy
from openarm_remote.robot_control.mod_arm import General_ArmController
from openarm_remote.robot_control.mod_ik import General_ArmIK
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from franka_msgs.action import Move ,Grasp 
from sensor_msgs.msg import JointState
from rclpy.action import ActionClient
from sensor_msgs.msg import Joy
import yaml
import os
from ament_index_python.packages import get_package_share_directory

class Robot(Node):
    def __init__(self):
        super().__init__("robot_node")   
        self.declare_parameter('save_dir', '~/default_save_path')
        self.declare_parameter('gripper_speed', 0.1)
        self.declare_parameter('gripper_force', 20.0)   
        self.t_position = None
        self.t_rotation = None
        self.table_position = None
        self.table_rotation = None
        package_share_directory = get_package_share_directory('openarm_remote')
        config_path = os.path.join(package_share_directory, 'config', 'robot_control.yaml')
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        # self.gripper = GripperController(self)
        
        self.active_arm = "left"  # 默认启动时控制左臂
        self.left_ik = General_ArmIK(config=full_config['robot_left_ik'])
        self.right_ik = General_ArmIK(config=full_config['robot_right_ik'])
        self.left_arm = General_ArmController(self,config=full_config['robot_left_arm'])
        self.right_arm = General_ArmController(self,config=full_config['robot_right_arm'])
        
        self.left_target_ee_pose = np.zeros(3)
        self.right_target_ee_pose = np.zeros(3)

        self.left_target_ee_rot = np.eye(3)
        self.right_target_ee_rot = np.eye(3)

        self.left_q = np.zeros(7)
        self.right_q = np.zeros(7)
        
    def get_observation(self):
        s = self.mod_arm.state
        q = s['position']
        ee_pose, ee_rot = self.left_ik.solve_fk(q)
        euler_angles = R.from_matrix(ee_rot).as_euler('xyz')
        # current_gripper_state = np.array([self.gripper.gripper_state ])
        # ee_pose_euler = np.concatenate([ee_pose, euler_angles, current_gripper_state])
        ee_pose_euler = np.concatenate([ee_pose, euler_angles, np.array([0])])
        # tag_tube = np.concatenate([self.t_position, self.t_rotation])
        # table = np.concatenate([self.table_position, self.table_rotation])
        return {
            # 'table': table,
            # 'tag_tube' : tag_tube,
            'timestamp': time.time(),
            'ee_pose': ee_pose,
            'ee_rot': ee_rot.flatten(),
            # **s,
            'position': s['position'],
            'action':  ee_pose_euler,
            # **self.camera_states
        }
    
    def step(self,arm_id: str, action):
        if len(action) != 6:
            raise ValueError("Action must be of length 6")
        if arm_id not in ["left", "right"]:
            raise ValueError("arm_id must be 'left' or 'right'")

        # 1. 根据 arm_id 选择要使用的对象和状态变量
        if arm_id == "left":
            ik_solver = self.left_ik
            arm_controller = self.left_arm
            target_pose = self.left_target_ee_pose
            target_rot = self.left_target_ee_rot
            q = self.left_q
        else:  # arm_id == "right"
            ik_solver = self.right_ik
            arm_controller = self.right_arm
            target_pose = self.right_target_ee_pose
            target_rot = self.right_target_ee_rot
            q = self.right_q
        # current_q = arm_controller.get_current_motor_q()
        # current_ee_pose, current_ee_rot = ik_solver.solve_fk(current_q)
        # if np.linalg.norm(target_pose - current_ee_pose) > 0.03:
        #     target_pose = current_ee_pose
        #     target_rot = current_ee_rot
            
        new_target_pose = target_pose + action[:3] * 0.005
        # new_target_pose = current_ee_pose + action[:3] * 0.005
        new_target_rot = (R.from_matrix(target_rot) * 
                        R.from_euler('xyz', action[3:6] * 0.01)).as_matrix()
        
        ee = np.eye(4)
        ee[:3, 3] = new_target_pose
        ee[:3, :3] = new_target_rot
        new_q, _ = ik_solver.solve_ik(ee, q)
        self.get_logger().info(f"Current {arm_id} joint positions: {new_q}", throttle_duration_sec=1.0)
        arm_controller.ctrl_dual_arm(new_q)
        # hand_msg = Joy()
        # hand_msg.axes = [0.0] * 7  # 初始化手部动作数组
        # hand_msg.axes[6] = action[6]  # 假设action的第7个元素是夹爪动作
        # self.gripper.state_pub.publish(hand_msg) #发布hand的topic消息,模拟手柄
        # self.hand_array[:] = action[6:]
        if arm_id == "left":
            self.left_target_ee_pose = new_target_pose
            self.left_target_ee_rot = new_target_rot
            self.left_q = new_q
        else:  # arm_id == "right"
            self.right_target_ee_pose = new_target_pose
            self.right_target_ee_rot = new_target_rot
            self.right_q = new_q
        return {
            'action.q': new_q,
            'action.ee_pose': new_target_pose,
        }
    
    def reset(self, arm_id: str,q=None, waite_time=1.5):
        if arm_id not in ["left", "right"]:
            raise ValueError("arm_id must be 'left' or 'right'")
        if arm_id == "left":
            arm_controller = self.left_arm_controller
            ik_solver = self.left_ik_solver
        else: # arm_id == "right"
            arm_controller = self.right_arm_controller
            ik_solver = self.right_ik_solver
            
        target_q = q
        if target_q is None:
            self._wait_for_joint_state()
            if arm_id == "left":
                target_q = self.left_q
            else: # arm_id == "right"
                target_q = self.right_q
        else:
            arm_controller.ctrl_dual_arm(target_q)
            target_pose, target_rot = ik_solver.solve_fk(target_q)
        if waite_time > 0:
            self.get_logger().info(f"Waiting for {waite_time} seconds for reset...")
            time.sleep(waite_time)#强行等待1.5秒    
        if arm_id == "left":
            self.left_q = target_q
            self.left_target_ee_pose = target_pose
            self.left_target_ee_rot = target_rot
        else: # arm_id == "right"
            self.right_q = target_q
            self.right_target_ee_pose = target_pose
            self.right_target_ee_rot = target_rot
        
        return {
            'action.q': target_q,
            'action.ee_pose': target_pose,
            'action.ee_rot': target_rot.flatten(),
        }

    @property
    def camera_states(self):
        wrist = self.wrist_camera.state
        front = self.front_camera.state
        return {
            'image.wrist.rgb': wrist['rgb'],
            'image.wrist.depth': wrist['depth'],
            'image.front.rgb': front['rgb'],
            'image.front.depth': front['depth']
        }

    def start(self):
        self.get_logger().info("Hand controller started")
        self._wait_for_joint_state()
        self.left_target_ee_pose, self.left_target_ee_rot = self.left_ik.solve_fk(self.left_q)
        self.right_target_ee_pose, self.right_target_ee_rot = self.right_ik.solve_fk(self.right_q)
    
    def _wait_for_joint_state(self):
        while rclpy.ok():
            self.get_logger().info(f"Waiting for  arm state...")
            self.left_q = self.left_arm.state['position']
            self.right_q = self.right_arm.state['position']
            if np.any(self.left_q) and np.any(self.right_q):
                self.get_logger().info(f"  Left Arm Initial Q: {self.left_q}")
                self.get_logger().info(f"  Right Arm Initial Q: {self.right_q}")
                
                # 3. 只有当两个手臂都准备好时，才跳出循环
                break 
            else:
                self.get_logger().info("No initial joint positions found, waiting...")
                time.sleep(0.03)
    
    def stop(self):
        self.get_logger().info("Hand controller stopped")