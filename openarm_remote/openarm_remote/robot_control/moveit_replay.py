#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
# MODIFIED: Import the new message type
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import json
import threading
import sys

class DirectRecordReplayNode(Node):
    def __init__(self):
        super().__init__('direct_record_replay_node_v2')

        # ### 1. ！！！Configuration Section (No changes needed here from before)！！！ ###
        self.control_topic = '/left_joint_trajectory_controller/joint_trajectory'
        self.joint_names = [
            'openarm_left_joint1', 'openarm_left_joint2', 'openarm_left_joint3',
            'openarm_left_joint4', 'openarm_left_joint5', 'openarm_left_joint6', 'openarm_left_joint7'
        ]
        # #######################################################

        # <<< MODIFIED SECTION START >>>
        # Subscribe to the controller's state topic
        state_topic = '/left_joint_trajectory_controller/state'
        self.subscription = self.create_subscription(
            JointTrajectoryControllerState, # New message type
            state_topic,
            self.controller_state_callback, # New callback function
            10)
        # <<< MODIFIED SECTION END >>>
        
        # The publisher part remains the same
        self.publisher = self.create_publisher(JointTrajectory, self.control_topic, 10)

        self.waypoints = []
        self.current_controller_state = None # Renamed for clarity
        self.lock = threading.Lock()

        self.get_logger().info("="*20 + " Direct Control Recorder V2 (using Controller State) " + "="*20)
        self.get_logger().info(f"Publishing commands to: '{self.control_topic}'")
        self.get_logger().info(f"Subscribing to state from: '{state_topic}'")
        # ... (Menu prints are the same)
        self.get_logger().info("  'r': Record, 'p': Play, 'c': Clear, 's': Save, 'l': Load, 'q': Quit")
        self.get_logger().info("="*78)

    # <<< MODIFIED SECTION START >>>
    def controller_state_callback(self, msg):
        """Callback to continuously update the current controller state."""
        with self.lock:
            self.current_controller_state = msg
    # <<< MODIFIED SECTION END >>>

    def record_waypoint(self):
        """Extracts and records a waypoint from the controller state."""
        with self.lock:
            if self.current_controller_state is None:
                self.get_logger().warn("Have not received any controller state messages yet. Cannot record.")
                return

            # <<< MODIFIED SECTION START >>>
            # The new message structure is simpler. The names and positions are already ordered.
            # We get the actual positions directly from the 'actual' field.
            # A quick sanity check is still good practice.
            if set(self.joint_names) != set(self.current_controller_state.joint_names):
                self.get_logger().error("FATAL: Joint names in script do not match joint names from controller state!")
                self.get_logger().error(f"Script names: {self.joint_names}")
                self.get_logger().error(f"Controller names: {self.current_controller_state.joint_names}")
                return
            
            # The actual positions are in msg.actual.positions
            waypoint = self.current_controller_state.actual.positions
            # <<< MODIFIED SECTION END >>>
            
            self.waypoints.append(list(waypoint)) # Convert to a plain list for JSON serialization
            self.get_logger().info(f"Waypoint recorded! Total waypoints: {len(self.waypoints)}.")

    # The replay and file functions do not need any changes, as they
    # only depend on the format of self.waypoints, which we have kept the same.
    def replay_trajectory(self):
        if not self.waypoints:
            self.get_logger().warn("Waypoint list is empty. Please record or load a path first.")
            return
        
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        
        time_from_start_sec = 2.0
        for waypoint in self.waypoints:
            point = JointTrajectoryPoint()
            point.positions = [float(p) for p in waypoint]
            point.time_from_start = Duration(sec=int(time_from_start_sec), nanosec=0)
            
            traj_msg.points.append(point)
            time_from_start_sec += 2.0

        self.get_logger().info("Publishing trajectory...")
        self.publisher.publish(traj_msg)
        self.get_logger().info("Trajectory published!")

    def save_waypoints_to_file(self):
        if not self.waypoints: self.get_logger().warn("No waypoints to save."); return
        filename = input("Enter filename to save (e.g., path.json): ")
        if not filename.endswith('.json'): filename += '.json'
        try:
            with open(filename, 'w') as f: json.dump(self.waypoints, f, indent=4)
            self.get_logger().info(f"Path saved to: {filename}")
        except Exception as e: self.get_logger().error(f"Failed to save file: {e}")

    def load_waypoints_from_file(self):
        filename = input("Enter filename to load (e.g., path.json): ")
        try:
            with open(filename, 'r') as f: self.waypoints = json.load(f)
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints from {filename}.")
        except FileNotFoundError: self.get_logger().error(f"Error: File '{filename}' not found.")
        except Exception as e: self.get_logger().error(f"Failed to load file: {e}")

    def clear_waypoints(self):
        self.waypoints = []
        self.get_logger().info("Waypoints cleared from memory.")

# The main loop also remains unchanged.
def main(args=None):
    rclpy.init(args=args)
    node = DirectRecordReplayNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        while rclpy.ok():
            choice = input("Enter command (r/p/c/s/l/q): ").lower()
            if choice == 'r': node.record_waypoint()
            elif choice == 'p': node.replay_trajectory()
            elif choice == 'c': node.clear_waypoints()
            elif choice == 's': node.save_waypoints_to_file()
            elif choice == 'l': node.load_waypoints_from_file()
            elif choice == 'q': break
            else: node.get_logger().warn("Invalid command.")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        node.get_logger().info("Shutting down...")
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join()

if __name__ == '__main__':
    main()