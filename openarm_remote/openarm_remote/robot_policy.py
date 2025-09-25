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

CAMERA_TOPIC = {
    'wrist': {
        'info': '/wrist/color/camera_info',
        'rgb': '/wrist/color/image_rect_raw/compressed',
        'depth': '/wrist/aligned_depth_to_color/image_raw/compressedDepth'
    },
    'front': {
        'info': '/front/zed_node/rgb/camera_info',
        'rgb': '/front/zed_node/rgb/image_rect_color/compressed',
        'depth': '/front/zed_node/depth/depth_registered/compressedDepth'
    }
}


class GripperController:
    def __init__(self, node: Node):
        self.node = node
        self.current_state = None  # 存储夹爪状态

        self.move_client = ActionClient(
            node=node,                # 第一个参数：节点实例
            action_type=Move,         # 第二个参数：action类型
            action_name="/fr3/franka_gripper/move"  # 第三个参数：action名称
        )
        self.grasp_client = ActionClient(
            node=node,
            action_type=Grasp,
            action_name= "/fr3/franka_gripper/grasp")
        
        # # 等待服务启动
        while not self.move_client.wait_for_server(timeout_sec=1.0):
            node.get_logger().info("等待夹爪move服务...")
        while not self.grasp_client.wait_for_server(timeout_sec=1.0):
            node.get_logger().info("等待夹爪grasp服务...")
        
        # 发布夹爪状态
        self.state_pub = node.create_publisher(
            Joy,
            "/joy_replay",
            10
        )
        node.get_logger().info("Franka夹爪控制器初始化完成")


    # def joy_callback(self, msg: Joy):
    #     """处理游戏手柄输入"""
    #     # Joy消息的axes是浮点数数组，使用数值比较而非字符串
    #     if msg.axes[6] == 1.0:  # 假设axes[6]为1时触发move
    #         self.node.get_logger().info("收到张开夹爪指令")
    #         self.move(width=0.08, speed=0.1)  # 张开到0.08米
    #     elif msg.axes[6] == -1.0:  # 假设axes[6]为-1时触发grasp
    #         self.node.get_logger().info("收到抓取指令")
    #         self.grasp(width=0.02, speed=0.1, force=30.0)

    def _state_callback(self, msg: JointState):
        self.current_state = msg

    # def open(self, speed: float = 0.1) -> bool:
    #     """默认张开到0.05米（预设值）"""
    #     return self.move(width=0.05, speed=speed)  # 固定宽度为0.05米

    def grasp(self, width=0.02, speed=0.1, force=30.0):
        goal = Grasp.Goal()
        goal.width = width
        goal.speed = speed
        goal.force = force
            # 检查动作客户端是否可用
        if not self.grasp_client.wait_for_server(timeout_sec=1.0):
            # self.node.get_logger().error("抓取动作服务器不可用")
            return False

        # 异步发送目标并添加回调
        self.node.get_logger().info(f"发送抓取指令: 宽度={width}, 速度={speed}, 力={force}")
        future = self.grasp_client.send_goal_async(
            goal,        )
        future.add_done_callback(self._grasp_done)
        return True
        
    def move(self, width=0.08, speed=0.1):
        goal = Move.Goal()
        goal.width = width
        goal.speed = speed
            # 检查动作客户端是否可用
        if not self.move_client.wait_for_server(timeout_sec=1.0):
            # self.node.get_logger().error("移动动作服务器不可用")
            return False

        # 异步发送目标并添加回调
        self.node.get_logger().info(f"发送移动指令: 宽度={width}, 速度={speed}")
        future = self.move_client.send_goal_async(
            goal,
        )
        future.add_done_callback(self._move_done)
        return True
    
         
class CameraNode:
    def __init__(self, node: Node, name: str = "wrist"):
        print("Initialize CameraNode...")
        self.node = node
        self.name = name
        self.rgb = b''
        self.depth = b''
        self.rgb_sub = node.create_subscription( #传入的name的字符串名
            CompressedImage,
            CAMERA_TOPIC[name]['rgb'],#从CAMERA_TOPIC字典中获取对应的RGB话题
            self._rgb_callback,
            10
        )
        self.depth_pub = node.create_subscription(
            CompressedImage,
            CAMERA_TOPIC[name]['depth'],
            self._depth_callback,
            10
        )
        print("Initialize CameraNode OK!\n")
    def _rgb_callback(self, msg: CompressedImage):
        self.rgb = msg.data.tobytes()

    def _depth_callback(self, msg: CompressedImage):
        self.depth = msg.data.tobytes()
    
    @property
    def state(self):
        return {
            'rgb': self.rgb,
            'depth': self.depth
        }    

class Robot(Node):
    def __init__(self):
        super().__init__("robot_node")
        self.declare_parameter('save_dir', 'recorder')
        self.declare_parameter('gripper_speed', 0.1)
        self.declare_parameter('gripper_force', 20.0)
        self.MID2_Q = np.array([0.129527314, 0.07156401,-0.42855033 ,-1.67398447 ,-0.06884267 ,1.6999068320,0.65245617])
        # self.MID2_Q = np.array([0.2700322, 0.37613487 ,-0.4840552,-1.12117557,0.0021750048574 ,0.8132924407543829,0.65245617])
        # 获取参数值
        save_dir = self.get_parameter('save_dir').value
        gripper_speed = self.get_parameter('gripper_speed').value
        gripper_force = self.get_parameter('gripper_force').value
        
        self.gripper = GripperController(self)
        
        self.mod_arm = General_ArmIK(self)
        self.mod_ik = General_ArmIK()

        self.target_ee_pose = np.zeros(3)
        self.target_ee_rot = np.eye(3)
        self.q = np.zeros(7)
        #测试的时候先不用摄像头了
        # self.wrist_camera = CameraNode(self, name="wrist")
        # self.front_camera = CameraNode(self, name="front")

    def get_observation(self):
        s = self.mod_arm.state
        q = s['position']
        ee_pose, ee_rot = self.mod_ik.solve_fk(q)
        gripper_width = self.gripper.current_state.width if self.gripper.current_state else 0.0
        return {
            'timestamp': time.time(),
            'ee_pose': ee_pose,
            'ee_rot': ee_rot.flatten(),
            # **s,
            'position': s['position'],
        }
    
    def step(self, action):
        if len(action) != 7: #回放时3+3+1
            raise ValueError("Action must be of length 7")

        self.target_ee_pose += action[:3]  # action[:3] * 0.005

        self.target_ee_rot = (R.from_matrix(self.target_ee_rot) *
                              R.from_euler('xyz', action[3:6] )).as_matrix() #欧拉角解旋转矩阵
        #R.from_euler('xyz', action[3:6] * 0.01)).as_matrix()
        ee = np.eye(4)#单位矩阵
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
    
    def reset(self, q=None, waite_time=1.0): #嵌套的函数传入q值
        if q is None:
            self._wait_for_joint_state()
        else:
            self.q = q
            self.mod_arm.ctrl_dual_arm(self.q)
            self.target_ee_pose, self.target_ee_rot = self.mod_ik.solve_fk(self.q)
        #这里夹爪就不松了,因为分段运行时第二次是夹钢管的,根据上一次运行完的最终态决定这一次
        # if hand_q is not None:
        #     self.hand_array[:] = hand_q
        if waite_time > 0:
            self.get_logger().info(f"Waiting for {waite_time} seconds for reset...")
            time.sleep(waite_time)
        return {
            'action.q': self.q,
            'action.ee_pose': self.target_ee_pose,
            'action.ee_rot': self.target_ee_rot.flatten(),
        }
    def reset_noMIDQ(self, q=None, waite_time=0.0): #嵌套的函数传入q值
        if q is None:
            self._wait_for_joint_state()
        else:
            self.q = q
            self.mod_arm.ctrl_dual_arm(self.q)
            self.target_ee_pose, self.target_ee_rot = self.mod_ik.solve_fk(self.q)
        #这里夹爪就不松了,因为分段运行时第二次是夹钢管的,根据上一次运行完的最终态决定这一次
        # if hand_q is not None:
        #     self.hand_array[:] = hand_q
        if waite_time > 0:
            self.get_logger().info(f"Waiting for {waite_time} seconds for reset...")
            time.sleep(waite_time)
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