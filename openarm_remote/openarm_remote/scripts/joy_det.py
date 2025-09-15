import rclpy
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import TransformStamped

from multiprocessing import Process, Array
import time
import threading
import os
import tf2_ros

import numpy as np
import jsonlines
import json

from openarm_remote.record.recorder import Recorder
from openarm_remote.robot import Robot
from openarm_remote.teleop.joystick import Joystick
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Joy 


KEYFRAME_FILE = get_package_share_directory('openarm_remote') + '/config/keyframes.jsonl'
KEYFRAMES = []
if os.path.exists(KEYFRAME_FILE):
    with jsonlines.open(KEYFRAME_FILE, 'r') as reader:
        for obj in reader:
            KEYFRAMES.append(obj)
    

def main(args=None):
    rclpy.init(args=args)
    #---
    node = Robot()
    joystick = Joystick(node)
    #---
    tf2_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf2_buffer, node)

    save_dir = node.get_parameter("save_dir").get_parameter_value().string_value
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    recorder = Recorder(save_dir)
    
    def waite_get_transform(frame_id, target_frame, timeout=1.0):
        """Wait for a transform to be available."""
        while rclpy.ok():
            try:
                node.get_logger().info(f"Waiting for transform {frame_id}->{target_frame}...")
                tf: TransformStamped = tf2_buffer.lookup_transform(
                    frame_id, target_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=timeout)
                )
                now = node.get_clock().now().to_msg()
                tf_time = tf.header.stamp
                dt = (rclpy.time.Time.from_msg(now) - rclpy.time.Time.from_msg(tf_time)).nanoseconds * 1e-9

                if dt > 0.5:
                    node.get_logger().warn(
                        f"TF too old (age={dt:.3f}s), waiting for a fresh one..."
                    )
                    time.sleep(0.1)
                    continue
                pos = np.array([tf.transform.translation.x,
                                tf.transform.translation.y,
                                tf.transform.translation.z])
                rot = np.array([tf.transform.rotation.x,
                                tf.transform.rotation.y,
                                tf.transform.rotation.z,
                                tf.transform.rotation.w])
                return pos, rot
            except tf2_ros.LookupException:
                node.get_logger().warn(f"Transform {frame_id}->{target_frame} not yet available")
                time.sleep(0.1)
    
    def teleop():
        node.start()
        rate = node.create_rate(30)
        q = np.zeros(7)
        recording = False
        keyframe = len(KEYFRAMES) - 1
        # tag_pose = np.zeros(3)
        # knuckle_pose = np.zeros(3)
        # tag_pose_cache = np.zeros(3)
        # knuckle_pose_cache = np.zeros(3)

        save_dir = node.get_parameter("save_dir").get_parameter_value().string_value
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        recorder = Recorder(save_dir)

        # tag_pose, _ = waite_get_transform("fr3_table_base", "tag_frame")
        # knuckle_pose, _ = waite_get_transform("fr3_table_base", "knuckle_frame")
        # node.get_logger().info(f"Initial tag pose: {tag_pose}, knuckle pose: {knuckle_pose}")
        # node.t_position, node.t_rotation = waite_get_transform(frame_id="fake_tag",target_frame="detected_target")
        # node.table_position, node.table_rotation = waite_get_transform(frame_id="fake_table_base",target_frame="table_base")
        
        while rclpy.ok():
            action = joystick.action #返回一个3+3+6的数组
            command = joystick.command

            try:
                action_obs = node.step(action[:6])  
            except Exception as e:
                pass

            # Start Recording
            if command[0] == 1:
                if not recording:
                    recorder.start()
                    # tag_pose_cache = tag_pose.copy()
                    # knuckle_pose_cache = knuckle_pose.copy()
                    recording = True
                    # node.get_logger().info(f"Recording started at tag pose: {tag_pose_cache}, knuckle pose: {knuckle_pose_cache}")
            # Stop Recording
            elif command[1] == 1:
                if recording:
                    recorder.stop()
                    recording = False
            # Reset Recording
            elif command[2] == 1:
                recorder.start()
                recording = False
            # Load Keyframe
            elif command[3] == 1:
                node.get_logger().info(f"Keyframe {keyframe} loaded.")
                # keyframe += 1
                # keyframe %= len(KEYFRAMES)
                # keyframe_data = KEYFRAMES[keyframe]
                q = np.array([0.11010828 , 0.11387246 ,-0.22371638 ,-2.10449131 ,-0.02231137 , 2.14695302,0.65245617])

                node.reset(q=q, waite_time=1.5)
                node.get_logger().info(f"Keyframe {keyframe} move finished.")
                rate.sleep()
                continue
            
            # if not any(q):
            #     rate.sleep()
            #     continue
            
            obs = node.get_observation()
            # action_obs = node.step(action[:6])
            # # action_obs = node.step(action)
            # # node.get_logger().info(f"当前action: {action}") 
            # obs["keyframe"] = KEYFRAMES[keyframe]['name']
            # obs["meta"] = {
            #     # "tag_pose": tag_pose_cache.tolist(),
            #     # "knuckle_pose": knuckle_pose_cache.tolist(),
            # }
            # obs = node.avro_datum
            # obs.update()
            if recording:
                recorder.record(obs)
            rate.sleep()
            
    thread = threading.Thread(target=teleop, daemon=True)
    thread.start()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        node.get_logger().info("Starting teleoperation with Multi-Threaded Executor.")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Teleoperation stopped by user.")
    finally:
        thread.join()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

    