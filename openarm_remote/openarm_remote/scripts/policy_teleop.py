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
    # policy = DetectPolicy(node, hand_array)
    policy = DetectPolicy(node, node.fr3_ik)
    save_dir = node.get_parameter("save_dir").get_parameter_value().string_value
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    recorder = Recorder(save_dir)
    
    def teleop():
        node.start()
        frame = 0 #这个应该是原tag分步的参数,根据不同记得改
        rate = node.create_rate(30)
        
        while rclpy.ok():
            obs = node.get_observation()#更新机械臂状态，机械臂当前状态返回的字典
            #传入的是写死的mid_q和手部动作，清空一下step_id，用于policy.step()获取动作,并且初始化机械臂位置
            if frame == 0 :
                node.reset_noMIDQ(np.array([0.11010828 , 0.11387246 ,-0.22371638 ,-2.10449131 ,-0.02231137 , 2.14695302,0.65245617]))
                time.sleep(1.0)
                action_obs = node.reset(**policy.reset(frame, obs))
                # recorder.start()
                # input("FRAME=0,Press Enter to start replay...")
            elif frame ==1 :
                node.reset_noMIDQ(node.MID2_Q)
                time.sleep(1.0)
                action_obs = node.reset_noMIDQ(**policy.reset_noMIDQ(frame, obs))
                # time.sleep(0.5)
            else :
                action_obs = node.reset_noMIDQ(**policy.reset_noMIDQ(frame, obs))
            obs.update(action_obs)
            
            policy.post_reset(obs)
            
            node.get_logger().info(f"Replay frame {frame} with {len(policy.actions)} actions.")
            # input("Press Enter to start replay...")
            
            while rclpy.ok() is not None:
                obs = node.get_observation()#更新机械臂状态，机械臂当前状态返回的字典
                data ,read_tag = policy.step(obs)#传入post_reset计算好的action
                if data is None:
                    break
                obs['tag_tube'] = read_tag
                obs_action = node.step(data)#发送action信息给话题
                obs['action'] = obs_action['action'][6:]
                # recorder.record(obs)
                rate.sleep()
            
            node.get_logger().info(f"Completed frame {frame}.")
            # input("Press Enter to continue to next frame...")
            frame += 1
            frame %= 4
            # if frame == 0:
                # recorder.stop()
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
