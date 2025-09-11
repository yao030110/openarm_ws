#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
import subprocess
import os
import time

class MoveItBagRecorder(Node):
    def __init__(self):
        super().__init__('moveit_bag_recorder')
        
        self.bag_process = None
        self.is_recording = False
        
        self.get_logger().info("MoveIt Bag Recorder 初始化完成")
    
    def start_recording(self, bag_name="moveit_trajectories"):
        """开始录制bag"""
        if self.is_recording:
            self.get_logger().warn("已经在录制中")
            return False
        
        try:
            # 使用ros2 bag record命令
            cmd = [
                'ros2', 'bag', 'record',
                '/left_joint_trajectory_controller/follow_joint_trajectory',
                '-o', bag_name
            ]
            self.bag_process = subprocess.Popen(cmd)
            self.is_recording = True
            self.get_logger().info(f"开始录制bag: {bag_name}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"开始录制时出错: {str(e)}")
            return False
    
    def stop_recording(self):
        """停止录制bag"""
        if self.bag_process and self.is_recording:
            self.bag_process.terminate()
            self.bag_process.wait()
            self.is_recording = False
            self.get_logger().info("停止录制bag")
            return True
        return False
    
    def play_bag(self, bag_name):
        """回放bag"""
        try:
            bag_path = os.path.join(os.getcwd(), bag_name)
            if not os.path.exists(bag_path):
                self.get_logger().error(f"Bag文件不存在: {bag_path}")
                return False
            
            cmd = ['ros2', 'bag', 'play', bag_name]
            subprocess.run(cmd, check=True)
            self.get_logger().info(f"回放bag: {bag_name}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"回放bag时出错: {str(e)}")
            return False

def main(args=None):
    rclpy.init(args=args)
    
    recorder = MoveItBagRecorder()
    
    try:
        # 开始录制
        recorder.start_recording("my_trajectories")
        
        # 在这里通过MoveIt界面或其他方式控制机械臂运动
        input("按回车键停止录制...")
        
        # 停止录制
        recorder.stop_recording()
        
        # 回放
        input("按回车键开始回放...")
        recorder.play_bag("my_trajectories")
        
        rclpy.spin(recorder)
        
    except KeyboardInterrupt:
        recorder.stop_recording()
    finally:
        recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()