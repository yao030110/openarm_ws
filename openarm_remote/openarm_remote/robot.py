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
        
        # self.gripper = GripperController(self)
        
        self.mod_arm = General_ArmController(self)
        self.mod_ik = General_ArmIK()

        self.target_ee_pose = np.zeros(3)
        self.target_ee_rot = np.eye(3)
        self.q = np.zeros(7)
        
    def get_observation(self):
        s = self.mod_arm.state
        q = s['position']
        ee_pose, ee_rot = self.mod_ik.solve_fk(q)
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
    
    def step(self, action):
        if len(action) != 6:
            raise ValueError("Action must be of length 6")

        self.target_ee_pose += action[:3] * 0.005

        self.target_ee_rot = (R.from_matrix(self.target_ee_rot) *
                              R.from_euler('xyz', action[3:6] * 0.01)).as_matrix()
        ee = np.eye(4)
        ee[:3, 3] = self.target_ee_pose
        ee[:3, :3] = self.target_ee_rot
        self.q, _ = self.mod_ik.solve_ik(ee, self.q)
        self.get_logger().info(f"Current joint positions: {self.q}", throttle_duration_sec=1.0)
        self.mod_arm.ctrl_dual_arm(self.q)
        # hand_msg = Joy()
        # hand_msg.axes = [0.0] * 7  # 初始化手部动作数组
        # hand_msg.axes[6] = action[6]  # 假设action的第7个元素是夹爪动作
        # self.gripper.state_pub.publish(hand_msg) #发布hand的topic消息,模拟手柄
        # self.hand_array[:] = action[6:]
        return {
            'action': action,
            'action.q': self.q,
            'action.ee_pose': self.target_ee_pose,
            'action.ee_rot': self.target_ee_rot.flatten(),
        }
    
    def reset(self, q=None, waite_time=1.5):
        if q is None:
            self._wait_for_joint_state()
        else:
            self.q = q
            self.mod_arm.ctrl_dual_arm(self.q)
            self.target_ee_pose, self.target_ee_rot = self.mod_ik.solve_fk(self.q)
        if waite_time > 0:
            self.get_logger().info(f"Waiting for {waite_time} seconds for reset...")
            time.sleep(waite_time)#强行等待1.5秒
        return {
            'action.q': self.q,
            'action.ee_pose': self.target_ee_pose,
            'action.ee_rot': self.target_ee_rot.flatten(),
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
        self.target_ee_pose, self.target_ee_rot = self.mod_ik.solve_fk(self.q)
    
    def _wait_for_joint_state(self):
        while rclpy.ok():
            self.get_logger().info("Waiting for arm state...")
            self.q = self.mod_arm.state['position']
            if not np.any(self.q):
                self.get_logger().info("No initial joint positions found, waiting...")
                time.sleep(0.03)
                self.get_logger().info("Retrying...")
            else:
                self.get_logger().info(f"Current joint positions: {self.q}")
                break
    
    def stop(self):
        self.get_logger().info("Hand controller stopped")