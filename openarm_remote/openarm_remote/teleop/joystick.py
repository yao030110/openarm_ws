import enum
import numpy as np
from sensor_msgs.msg import Joy
import threading
from rclpy.node import Node

class Axis(enum.Enum):
    LEFT_STICK_X = 0
    LEFT_STICK_Y = 1
    LEFT_TRIGGER = 2
    RIGHT_STICK_X = 3
    RIGHT_STICK_Y = 4
    RIGHT_TRIGGER = 5
    D_PAD_X = 6
    D_PAD_Y = 7

class Button(enum.Enum):
    A = 0
    B = 1
    X = 2
    Y = 3
    LEFT_BUMPER = 4
    RIGHT_BUMPER = 5
    CHANGE_VIEW = 6
    MENU = 7
    HOME = 8
    LEFT_STICK_CLICK = 9
    RIGHT_STICK_CLICK = 10
    
HAND_MODE = [0,1,2,3]

class Joystick:
    def __init__(self, node: Node):
        self.node = node
        self.joystick_sub = node.create_subscription(
            Joy,
            "/joy",
            self.joystick_callback,
            10
        )
        self.hand_action = HAND_MODE[0]
        self.command_lock = threading.Lock()
        self._command = [0, 0, 0, 0]  # A, B, X, Y buttons
        self.action_lock = threading.Lock()
        self._action = np.zeros(6 + 1)  # 6 for arm control, 6 for hand control
        self._action[6] = self.hand_action
        self.last_button_press_time = 0.0
        
    def joystick_callback(self, msg:Joy):
        z = msg.axes[Axis.RIGHT_STICK_Y.value]
        y = msg.axes[Axis.RIGHT_STICK_X.value]
        lin_x_right = -0.5 * (msg.axes[Axis.RIGHT_TRIGGER.value] - 1)
        lin_x_left = 0.5 * (msg.axes[Axis.LEFT_TRIGGER.value] - 1)
        x = lin_x_right + lin_x_left
        ang_y = msg.axes[Axis.LEFT_STICK_Y.value]
        ang_x = msg.axes[Axis.LEFT_STICK_X.value]
        roll_positive = msg.buttons[Button.LEFT_BUMPER.value]
        roll_negative = -msg.buttons[Button.RIGHT_BUMPER.value]
        ang_z = roll_positive + roll_negative
        if msg.axes[Axis.D_PAD_X.value] > 0:
            self.hand_action = HAND_MODE[0]
        elif msg.axes[Axis.D_PAD_X.value] < 0:
            self.hand_action = HAND_MODE[1]
        elif msg.axes[Axis.D_PAD_Y.value] > 0:
            self.hand_action = HAND_MODE[2]
        elif msg.axes[Axis.D_PAD_Y.value] < 0:
            self.hand_action = HAND_MODE[3]
        
        with self.action_lock:
            self._action[:3] = [x, y, z]
            self._action[3:6] = [ang_x, ang_y, ang_z]
            self._action[6:] = self.hand_action
        
        with self.command_lock:
            self._command = [msg.buttons[Button.A.value], msg.buttons[Button.B.value], msg.buttons[Button.X.value], msg.buttons[Button.Y.value]]
    @property
    def action(self):
        with self.action_lock:
            return self._action.copy()
    
    @property
    def command(self):
        with self.command_lock:
            return self._command.copy()
    def set_hand_action6_value(self, hand_action_value: int):
        with self.action_lock:
            # 在锁的保护下，安全地修改内部状态
            # 这里我们假设您只想修改第7个元素(索引6)
            self._action[6] = hand_action_value