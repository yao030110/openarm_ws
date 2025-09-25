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
    arm_policy = DetectPolicy(node, node.mod_ik)
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
            arm_actions = np.array([])

            # --- 处理左臂 ---
            try:
                node.get_logger().info(f"Planning for LEFT arm, frame {frame}...")
                action_obs = node.reset(**arm_policy.reset_noMIDQ(frame, obs)) # 可能会因文件不存在而报错
                action_obs.update(obs)
                arm_actions = arm_policy.post_reset(action_obs)
            except FileNotFoundError:
                # 捕获错误，打印警告，此时 left_actions 仍然是空列表
                node.get_logger().warn(f"No replay file found for LEFT arm, frame {frame}. It will hold position.")
            except Exception as e:
                # 捕获其他可能的规划错误
                node.get_logger().error(f"An unexpected error occurred while planning for LEFT arm: {e}")

            node.get_logger().info(f"Replay frame {frame}. Arm: {len(arm_actions)} actions")
        
            
            # --- 2. 执行阶段 (在这里实现多线程同步) ---

            # 定义一个局部的“工作函数”，负责播放单个手臂的轨迹
            def replay_worker( actions):
                for action in actions:
                    if not rclpy.ok():
                        break
                    try:
                        node.step( action=action)
                    except Exception as e:
                        node.get_logger().error(f"Error in step for arm: {e}")
                        break
                    rate.sleep()
                    
            input("Press Enter to continue...") # 如果需要可以取消注释
            # 为左右臂分别创建工作线程
            arm_thread = threading.Thread(target=replay_worker, args=( arm_actions))

            # 同时启动两个线程
            arm_thread.start()

            # 【关键】主线程在这里等待，直到两个工作线程都执行完毕
            arm_thread.join()

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
