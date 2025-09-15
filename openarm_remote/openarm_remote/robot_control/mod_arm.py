import numpy as np
import threading
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from rclpy.node import Node
import rclpy
import yaml
import os

class General_ArmController:
    def __init__(self, node: Node):
        print("Initialize ArmController...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        package_root_dir = os.path.dirname(script_dir)
        config_path = os.path.join(package_root_dir, 'config', 'robot_control.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.robot_arm = config['robot_for_arm']
        
        self.node = node
        self.arm_velocity_limit = np.array(self.robot_arm['velocity_limits'])
        self.control_dt = 1.0 / self.robot_arm['control_frequency_hz']
        self.states = {
            'position': np.zeros(7),
        }
        self.joint_state_sub = node.create_subscription(
            JointState, 
            self.robot_arm['joint_state_sub'], 
            self._joint_state_callback, 
            10
        )
        self.command_pub = node.create_publisher(
            JointState, 
            self.robot_arm['joint_targer_pub'], 
            10
        )
        self.q_target = np.zeros(7)
        self.ctrl_lock = threading.Lock()
        self.states_lock = threading.Lock()
        self._init=False
        self.thread = threading.Thread(target=self._ctrl_arm, daemon=True)
        self.thread.start()
        print("Initialize General_ArmController OK!\n")

    def _joint_state_callback(self, msg: JointState):
        with self.states_lock:
            self.states['position'] = np.array(msg.position)
            
    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.states['position'].copy()
        delta = target_q - current_q
        motion_scale = np.abs(delta) / (velocity_limit * self.control_dt)
        scale = motion_scale.max()
        cliped_arm_q_target = current_q + delta / max(scale, 1.0)
        return cliped_arm_q_target
    
    def _ctrl_arm(self):
        rate = self.node.create_rate(1.0 / self.control_dt)
        while rclpy.ok():
            if not any(self.states['position']):
                print("Waiting for initial arm state...")
                rate.sleep()
                continue
            if not self._init:
                self._init = True
                with self.ctrl_lock:
                    self.q_target = self.states['position'].copy()
            with self.ctrl_lock:
                arm_q_target = self.q_target

            cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)
            command = JointState()
            command.position = cliped_arm_q_target.tolist()
            self.command_pub.publish(command)
            rate.sleep()
    
    def ctrl_dual_arm(self, q_target):
        with self.ctrl_lock:
            self.q_target = q_target
            
    def get_mode_machine(self):
        pass
    
    def get_current_motor_q(self):
        return self.states['position'].copy()
    
    
    def ctrl_arm_go_home(self):
        pass
    
    def speed_gradual_max(self, t = 5.0):
        pass
    
    def speed_instant_max(self):
        pass
    
    @property
    def state(self):
        with self.states_lock:
            return self.states.copy()