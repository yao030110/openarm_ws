import rclpy
import time
import os
from rclpy.executors import MultiThreadedExecutor
import threading
from multiprocessing import Process, Array
from openarm_remote.record.recorder import Recorder
from openarm_remote.policy.det_policy import DetectPolicy
from openarm_remote.robot_policy import Robot
import numpy as np
def main(args=None, hand_array=None, hand_state=None):
    rclpy.init(args=args)
    node = Robot()
    # policy = DetectPolicy(node, node.fr3_ik)
    left_policy = DetectPolicy(node, node.left_ik ,arm_id="left")
    right_policy = DetectPolicy(node, node.right_ik , arm_id="right")
    save_dir = node.get_parameter("save_dir").get_parameter_value().string_value
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    recorder = Recorder(save_dir)
    
    def teleop():
        node.start()
        frame = 0 #这个应该是原tag分步的参数,根据不同记得改
        rate = node.create_rate(30)
        
        while rclpy.ok():
            obs = node.get_observation()
            
           # 先为两个手臂的动作序列准备好空的默认值
            left_actions = np.array([])
            right_actions = np.array([])

            # --- 处理左臂 ---
            try:
                node.get_logger().info(f"Planning for LEFT arm, frame {frame}...")
                action_left_obs = node.reset(**left_policy.reset_noMIDQ(frame, obs)) # 可能会因文件不存在而报错
                action_left_obs.update(obs['left'])
                left_actions = left_policy.post_reset(action_left_obs)
            except FileNotFoundError:
                # 捕获错误，打印警告，此时 left_actions 仍然是空列表
                node.get_logger().warn(f"No replay file found for LEFT arm, frame {frame}. It will hold position.")
            except Exception as e:
                # 捕获其他可能的规划错误
                node.get_logger().error(f"An unexpected error occurred while planning for LEFT arm: {e}")

            # --- 处理右臂 ---
            try:
                node.get_logger().info(f"Planning for RIGHT arm, frame {frame}...")
                action_right_obs = node.reset(**right_policy.reset_noMIDQ(frame, obs)) # 可能会因文件不存在而报错
                action_right_obs.update(obs['right'])
                right_actions = right_policy.post_reset(action_right_obs)
            except FileNotFoundError:
                node.get_logger().warn(f"No replay file found for RIGHT arm, frame {frame}. It will hold position.")
            except Exception as e:
                node.get_logger().error(f"An unexpected error occurred while planning for RIGHT arm: {e}")

            node.get_logger().info(f"Replay frame {frame}. Left: {len(left_actions)} actions, Right: {len(right_actions)} actions.")
        
            
            # --- 2. 执行阶段 (在这里实现多线程同步) ---

            # 定义一个局部的“工作函数”，负责播放单个手臂的轨迹
            def replay_worker(arm_id, actions):
                for action in actions:
                    if not rclpy.ok():
                        break
                    try:
                        node.step(arm_id=arm_id, action=action)
                    except Exception as e:
                        node.get_logger().error(f"Error in step for {arm_id} arm: {e}")
                        break
                    rate.sleep()
                    
            input("Press Enter to continue...") # 如果需要可以取消注释
            # 为左右臂分别创建工作线程
            left_thread = threading.Thread(target=replay_worker, args=("left", left_actions))
            right_thread = threading.Thread(target=replay_worker, args=("right", right_actions))

            # 同时启动两个线程
            left_thread.start()
            right_thread.start()

            # 【关键】主线程在这里等待，直到两个工作线程都执行完毕
            left_thread.join()
            right_thread.join()
            
            # --- 3. 完成阶段 ---
            node.get_logger().info(f"--- Completed frame {frame} by both arms. ---")
            
            # input("Press Enter to continue...") # 如果需要可以取消注释
            
            # 进入下一个任务片段
            frame = (frame + 1) % 4
    thread = threading.Thread(target=teleop, daemon=True)
    thread.start()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()

if __name__ == "__main__":
    # hand_array = Array('d', 6, lock=True) #共享内存
    # hand_state = Array('d', 6, lock=True)
    # p = Process(target=run_brainco_controller, args=(hand_array, hand_state,))
    # p.start()#启动进程

    # main(hand_array=hand_array, hand_state=hand_state)
    main()#直接运行teleop
    # p.join()#确保子进程完成再结束程序,当主进程调用 join() 后，主进程会等待子进程结束，再执行退出操作。
