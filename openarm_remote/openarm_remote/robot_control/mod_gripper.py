# ros2 action send_goal /left_gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{position: X, max_effort: Y}"
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
import threading

class GripperController:
    """
    一个通用的、可重用的夹爪控制器类。
    采用【阻塞式】初始化，确保在对象创建成功后，控制器立刻可用。
    """
    def __init__(self, node: Node, config: dict, timeout_sec: float = 5.0):
        self.node = node
        self.config = config
        self.logger = self.node.get_logger()

        action_name = self.config['action_name']
        
        self.logger.info(f"Creating Gripper Action Client for '{action_name}'...")
        self._action_client = ActionClient(self.node, GripperCommand, action_name)

        # --- 关键修改：直接在这里阻塞等待 ---
        self.logger.info(f"Waiting for gripper server '{action_name}' for up to {timeout_sec} seconds...")
        server_ready = self._action_client.wait_for_server(timeout_sec=timeout_sec)
        
        # 如果在超时时间内没有等到服务器，则抛出异常，让整个程序启动失败
        if not server_ready:
            self.logger.error(f"Action server '{action_name}' not available after waiting.")
            raise RuntimeError(f"Gripper action server '{self.config['action_name']}' not available.")
            
        self.logger.info(f"Gripper Action Server '{self.config['action_name']}' is connected and ready.")
        
        # --- 状态变量 (依然需要线程锁，因为回调函数会在不同线程中执行) ---
        self._state_lock = threading.Lock()
        self._last_result = None
        self._last_feedback = None

    def send_goal(self, position: float, max_effort: float):
        # --- 关键修改：不再需要 is_ready() 判断 ---
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort

        self.logger.info(f"Sending goal to '{self.config['action_name']}': pos={position:.3f}m, effort={max_effort:.1f}N")
        
        future = self._action_client.send_goal_async(goal_msg, feedback_callback=self._feedback_callback)
        future.add_done_callback(self._goal_response_callback)

    def open(self, max_effort: float = None):
        """发送一个完全打开夹爪的指令。"""
        if max_effort is None:
            max_effort = self.config['default_max_effort']
        self.send_goal(self.config['open_position'], max_effort)

    def close(self, max_effort: float = None):
        """发送一个完全闭合夹爪的指令。"""
        if max_effort is None:
            max_effort = self.config['default_max_effort']
        self.send_goal(self.config['closed_position'], max_effort)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.logger.warn(f"Gripper goal for '{self.config['action_name']}' was rejected.")
            return
        
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._get_result_callback)

    def _get_result_callback(self, future):
        with self._state_lock:
            self._last_result = future.result().result
        
        self.logger.info(f"Gripper '{self.config['action_name']}' goal finished with result: Stalled={self._last_result.stalled}")

    def _feedback_callback(self, feedback_msg):
        with self._state_lock:
            self._last_feedback = feedback_msg.feedback
        
        # self.logger.info(f"Gripper feedback: {self._last_feedback.position}", throttle_duration_sec=1.0)
    
    @property
    def stalled(self) -> bool:
        """如果夹爪因为夹住物体而停止，返回True。"""
        with self._state_lock:
            if self._last_result:
                return self._last_result.stalled
        return False
        
    @property
    def current_position(self) -> float:
        """返回夹爪当前的开口宽度。"""
        with self._state_lock:
            if self._last_feedback:
                return self._last_feedback.position
        return 0.0
    
if __name__ == "__main__":
    # 1. 导入【仅用于测试的】库
    import sys
    import termios
    import tty
    import select
    import os
    import yaml
    from ament_index_python.packages import get_package_share_directory
    import time

    # 2. 定义【仅用于测试的】非阻塞键盘监听工具
    class KBHit:
        def __init__(self):
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setraw(sys.stdin.fileno())
        def __del__(self):
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        def kbhit(self):
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
        def getch(self):
            return sys.stdin.read(1)

    # 3. 定义【仅用于测试的】主函数
    def main(args=None):
        rclpy.init(args=args)
        # 创建一个临时的ROS节点用于测试
        test_node = Node("gripper_test_node")

        try:
            # --- 加载配置文件 ---
            package_share_directory = get_package_share_directory('openarm_remote')
            config_path = os.path.join(package_share_directory, 'config', 'robot_control.yaml')
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            # 提取左臂夹爪的配置
            left_gripper_config = full_config['left_gripper_config']
            
            # --- 初始化左臂夹爪控制器 ---
            # __init__会阻塞等待服务器，如果成功，则gripper立即可用
            left_gripper = GripperController(test_node, config=left_gripper_config)
            
            # --- 打印操作说明 ---
            print("\n" + "="*50)
            print("Gripper Interactive Test")
            print("Press keys to control the LEFT gripper.")
            print("="*50)
            print("  o: Open gripper")
            print("  c: Close gripper")
            print("  q: Quit")
            print("="*50 + "\n")
            kb = KBHit()
            # 启动一个独立的线程来spin节点，以便Action回调可以被处理
            executor_thread = threading.Thread(target=rclpy.spin, args=(test_node,), daemon=True)
            executor_thread.start()

            # --- 进入主循环，监听键盘 ---
            while True:
                if kb.kbhit():
                    key = kb.getch()
                    if key == 'o' or key == 'O':
                        print("Command: OPEN")
                        left_gripper.open()
                    elif key == 'c' or key == 'C':
                        print("Command: CLOSE")
                        left_gripper.close()
                    elif key == 'a' or key == 'A':
                        print("Command: do you self")
                        left_gripper.send_goal(0.02,20.0)
                    elif key == 'q' or key == 'Q':
                        print("Exiting...")
                        break
                if time.time() % 0.5 < 0.1:  # 每半秒打印一次（简化版）
                    pos = left_gripper.current_position
                    stalled = left_gripper.stalled
                    print(f"\rPosition: {pos:.3f}m | Stalled: {stalled}", end="", flush=True)
                time.sleep(0.1) # 10Hz 循环

        except RuntimeError as e:
            test_node.get_logger().fatal(f"Initialization failed: {e}")
        except KeyboardInterrupt:
            pass
        finally:
            test_node.get_logger().info("Shutting down.")
            test_node.destroy_node()
            rclpy.shutdown()

    # --- 执行主测试函数 ---
    main()