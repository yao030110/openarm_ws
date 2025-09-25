from rclpy.node import Node
import rclpy
from openarm_remote.robot_control.mod_arm import General_ArmController
from openarm_remote.robot_control.mod_ik import General_ArmIK
from openarm_remote.robot_control.mod_gripper import GripperController
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
        self.declare_parameter('save_dir', 'recordings')
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
        
        
        self.active_arm = "left"  # 默认启动时控制左臂
        self.left_gripper = GripperController(self,config=full_config['left_gripper_config'])
        self.right_gripper = GripperController(self,config=full_config['right_gripper_config'])
        self.left_ik = General_ArmIK(config=full_config['robot_left_ik'])
        self.right_ik = General_ArmIK(config=full_config['robot_right_ik'])
        self.left_arm = General_ArmController(self,config=full_config['robot_left_arm'])
        self.right_arm = General_ArmController(self,config=full_config['robot_right_arm'])
        self.last_left_gripper_command = -1
        self.last_right_gripper_command = -1
        
        self.left_target_ee_pose = np.zeros(3)
        self.right_target_ee_pose = np.zeros(3)

        self.left_target_ee_rot = np.eye(3)
        self.right_target_ee_rot = np.eye(3)

        self.left_q = np.zeros(7)
        self.right_q = np.zeros(7)
        
    def get_observation(self) -> dict:
        """【最终版】一次性获取双臂的完整状态。"""
        left_q = self.left_arm.get_current_motor_q()
        left_ee_pose, left_ee_rot = self.left_ik.solve_fk(left_q)
        right_q = self.right_arm.get_current_motor_q()
        right_ee_pose, right_ee_rot = self.right_ik.solve_fk(right_q)
        
        return {
            'timestamp': time.time(),
            'left': {
                'joint_positions': left_q, 'ee_pose': left_ee_pose, 'ee_rot_flat': left_ee_rot.flatten(),
                'gripper_position': self.left_gripper.current_position, 'gripper_stalled': self.left_gripper.stalled
            },
            'right': {
                'joint_positions': right_q, 'ee_pose': right_ee_pose, 'ee_rot_flat': right_ee_rot.flatten(),
                'gripper_position': self.right_gripper.current_position, 'gripper_stalled': self.right_gripper.stalled
            }
        }
    
    def step(self,action , arm_id: str = None ):
        if len(action) != 7:
            raise ValueError("Action must be of length 7")
        if arm_id not in ["left", "right"]:
            raise ValueError("arm_id must be 'left' or 'right'")

        # 1. 根据 arm_id 选择要使用的对象和状态变量
        if arm_id == "left":
            ik_solver, arm_controller, grip = self.left_ik, self.left_arm, self.left_gripper
            target_pose = self.left_target_ee_pose
            target_rot = self.left_target_ee_rot
            q = self.left_q
            last_gripper_command = self.last_left_gripper_command
        else:  # arm_id == "right"
            ik_solver, arm_controller, grip = self.right_ik, self.right_arm, self.right_gripper
            target_pose = self.right_target_ee_pose
            target_rot = self.right_target_ee_rot
            q = self.right_q
            last_gripper_command = self.last_right_gripper_command
        # current_q = arm_controller.get_current_motor_q()
        # current_ee_pose, current_ee_rot = ik_solver.solve_fk(current_q)
        # if np.linalg.norm(target_pose - current_ee_pose) > 0.03:
        #     target_pose = current_ee_pose
        #     target_rot = current_ee_rot
            
        new_target_pose = target_pose + action[:3] 
        # new_target_pose = current_ee_pose + action[:3] * 0.005
        new_target_rot = (R.from_matrix(target_rot) * 
                        R.from_euler('xyz', action[3:6] * 0.01)).as_matrix()
        
        ee = np.eye(4)
        ee[:3, 3] = new_target_pose
        ee[:3, :3] = new_target_rot
        new_q, _ = ik_solver.solve_ik(ee, q)
        self.get_logger().info(f"Current {arm_id} joint positions: {new_q}", throttle_duration_sec=1.0)
        arm_controller.ctrl_dual_arm(new_q)
        current_gripper_command = action[6] # 获取当前的夹爪指令 (0或1)

        # 检查指令是否从上一帧发生了“变化”
        if current_gripper_command != last_gripper_command:
            self.get_logger().info(f"Gripper command changed for '{arm_id}' arm to: {current_gripper_command}")
            # 根据变化后的新值，执行一次性的 open() 或 close()
            if current_gripper_command == 1:
                grip.open()
            elif current_gripper_command == 0:
                grip.close()
        if arm_id == "left":
            self.left_target_ee_pose = new_target_pose
            self.left_target_ee_rot = new_target_rot
            self.left_q = new_q
            self.last_left_gripper_command = current_gripper_command
        else:  # arm_id == "right"
            self.right_target_ee_pose = new_target_pose
            self.right_target_ee_rot = new_target_rot
            self.right_q = new_q
            self.last_right_gripper_command = current_gripper_command
        return {
            'action.q': new_q,
            'action.ee_pose': new_target_pose,
        }
    
    def reset(self, arm_id: str = None,q=None, waite_time=1.5):
        if arm_id not in ["left", "right"]:
            raise ValueError("arm_id must be 'left' or 'right'")
        if arm_id == "left":
            arm_controller = self.left_arm
            ik_solver = self.left_ik
        else: # arm_id == "right"
            arm_controller = self.right_arm
            ik_solver = self.right_ik
            
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
        if arm_id == "left":
            self.left_q = target_q
            self.left_target_ee_pose = target_pose
            self.left_target_ee_rot = target_rot
        else: # arm_id == "right"
            self.right_q = target_q
            self.right_target_ee_pose = target_pose
            self.right_target_ee_rot = target_rot
        if waite_time > 0:
            self.get_logger().info(f"Waiting for {waite_time} seconds for reset...")
            self.get_logger().info(f"arm_id is {arm_id} ")
            time.sleep(waite_time)#强行等待1.5秒  

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