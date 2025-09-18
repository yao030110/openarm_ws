import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
import yaml
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer
from ament_index_python.packages import get_package_share_directory
from openarm_remote.utils.weighted_moving_filter import WeightedMovingFilter
import os

class General_ArmIK:
    def __init__(self, config: dict ,Visualization=False, filter=False ):
        # __file__ 是当前脚本(mod_ik.py)的路径
       
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.filter = filter
        self.Visualization = Visualization
        self.robot_ik = config
        assets_package_path = get_package_share_directory(self.robot_ik['package_name'])
        urdf_full_path = assets_package_path + self.robot_ik['urdf_path']
        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_full_path,
            assets_package_path 
        )
        # self.mixed_jointsToLockIDs = []
        locked_joint_names = self.robot_ik.get('locked_joints') 

        # 检查获取到的值是否为None或者是一个空列表
        if locked_joint_names:
            self.mixed_jointsToLockIDs = [
                self.robot.model.getJointId(joint_name) for joint_name in locked_joint_names
            ]
        else:
            # 如果是None或空列表，直接赋值为空列表
            self.mixed_jointsToLockIDs = []

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        for idx, name in enumerate(self.reduced_robot.model.names):
            print(f"{idx}: {name}")
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.hand_id = self.reduced_robot.model.getFrameId(self.robot_ik['end_effector_frame'])

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.hand_id].translation - self.cTf[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.hand_id].rotation @ self.cTf[:3, :3].T),
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf))
        # self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost +
                           self.rotation_cost + 0.1 * self.smooth_cost)

        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 100,
                'tol': 1e-6
                # 'tol': 1e-5
            },
            'print_time': False,  # print or not
            # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
            'calc_lam_p': False
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 7)
        self.vis = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model, 
                self.reduced_robot.collision_model, 
                self.reduced_robot.visual_model
                )
            self.vis.initViewer(open=False)
            self.vis.loadViewerModel("pinocchio")
            self.vis.displayFrames(
                True, frame_ids=[0, self.hand_id], axis_length=0.15, axis_width=5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = [self.robot_ik['end_effector_frame']]
            FRAME_AXIS_POSITIONS = (
                np.array([[0, 0, 0], [1, 0, 0],
                          [0, 0, 0], [0, 1, 0],
                          [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            FRAME_AXIS_COLORS = (
                np.array([[1, 0, 0], [1, 0.6, 0],
                          [0, 1, 0], [0.6, 1, 0],
                          [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            )
            axis_length = 0.1
            axis_width = 20
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )
    # If the robot arm is not the same size as your arm :)

    # def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
    #     scale_factor = robot_arm_length / human_arm_length
    #     robot_left_pose = human_left_pose.copy()
    #     robot_right_pose = human_right_pose.copy()
    #     robot_left_pose[:3, 3] *= scale_factor
    #     robot_right_pose[:3, 3] *= scale_factor
    #     return robot_left_pose, robot_right_pose

    def solve_ik(self, wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        # 在求解前先检测目标是否可达
        # if not self.is_target_reachable(wrist):
        #     # 目标不可达，进行限制或警告
        #     wrist = self.clamp_target_to_workspace(wrist)
        #     print("Warning: Target outside workspace, clamping to boundary")

        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        if self.Visualization:
            self.vis.viewer[self.robot_ik['end_effector_frame']].set_transform(wrist)   # for visualization
        self.opti.set_value(self.param_tf, wrist)
        self.opti.set_value(self.var_q_last, self.init_data)  # for smooth
        try:
            sol = self.opti.solve()
            sol_q = self.opti.value(self.var_q)
            if self.filter:
                self.smooth_filter.add_data(sol_q)
                sol_q = self.smooth_filter.filtered_data
            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0
            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data,
                                 sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            
            if self.filter:
                self.smooth_filter.add_data(sol_q)
                sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data,
                                 sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)

    def solve_fk(self, q):
        """
        Perform forward kinematics to compute the end-effector pose.
        :param q: Joint configuration
        :return: End-effector pose as a pinocchio SE3 object
        """
        rt = self.reduced_robot.framePlacement(q, self.hand_id)
        return rt.translation, rt.rotation


if __name__ == "__main__":
    # from scipy.spatial.transform import Rotation as R
    # package_share_directory = get_package_share_directory('openarm_remote')
    # config_path = os.path.join(package_share_directory, 'config', 'robot_control.yaml')
    # with open(config_path, 'r') as f:
    #     full_config = yaml.safe_load(f)
    # ik = General_ArmIK(Visualization=True,config=full_config['robot_right_ik'])
    from scipy.spatial.transform import Rotation as R
    import math
    import sys
    import termios
    import tty
    import select
    
    # 2. 把【仅用于测试的】辅助类定义在这里
    class KBHit:
        """
        一个用于非阻塞式键盘监听的类。
        它的所有逻辑都封装在 __main__ 块内，不会影响外部导入。
        """
        def __init__(self):
            # 保存终端的原始设置
            self.fd = sys.stdin.fileno()
            try:
                self.old_settings = termios.tcgetattr(self.fd)
                # 设置终端为原始模式
                tty.setraw(sys.stdin.fileno())
            except termios.error:
                print("Warning: Not a TTY, keyboard input will not work.")
                self.old_settings = None

        def __del__(self):
            # 恢复终端的原始设置
            if self.old_settings:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

        def kbhit(self):
            # 检查是否有按键事件
            if self.old_settings:
                return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
            return False

        def getch(self):
            # 获取按下的单个字符
            if self.old_settings:
                return sys.stdin.read(1)
            return ''

    # 3. 把【仅用于测试的】主逻辑放在这里
    def main_interactive_test():
        # --- 初始化 IK 求解器 ---
        package_share_directory = get_package_share_directory('openarm_remote')
        config_path = os.path.join(package_share_directory, 'config', 'robot_control.yaml')
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        ik = General_ArmIK(config=full_config['robot_left_ik'], Visualization=True)

        # --- 定义控制参数 ---
        LINEAR_STEP = 0.01  # 每次按键平移的距离 (米)
        ANGULAR_STEP = 0.05 # 每次按键旋转的角度 (弧度, 约3度)
        LOOP_RATE = 30      # Hz, 控制循环频率

        # --- 定义一个安全的初始位姿 ---
        start_pos = np.array([0.4, 0.0, 0.4])
        start_rot_matrix = R.from_euler('y', -np.pi / 2).as_matrix()
        tf_target = pin.SE3(start_rot_matrix, start_pos)

        # --- 打印操作说明 ---
        print("\n" + "="*50)
        print("Interactive IK Control Started.")
        print("Press keys to move the end-effector. You do not need to press Enter.")
        print("="*50)
        print("  --- Translation ---       --- Rotation ---")
        print("    w/s: +X / -X             u/j: +Pitch / -Pitch (Y-axis)")
        print("    a/d: +Y / -Y             h/k: +Yaw / -Yaw (Z-axis)")
        print("    q/e: +Z / -Z             y/i: +Roll / -Roll (X-axis)")
        print("\n    x: Quit")
        print("="*50 + "\n")

        # --- 初始化并进入主循环 ---
        kb = KBHit()
        
        print("Moving to initial pose...")
        ik.solve_ik(tf_target.homogeneous)
        time.sleep(2.0)
        print("Ready to control.")

        try:
            while True:
                if kb.kbhit():
                    key = kb.getch()

                    # 更新目标位姿
                    if key == 'w': tf_target.translation[0] += LINEAR_STEP
                    elif key == 's': tf_target.translation[0] -= LINEAR_STEP
                    elif key == 'a': tf_target.translation[1] += LINEAR_STEP
                    elif key == 'd': tf_target.translation[1] -= LINEAR_STEP
                    elif key == 'q': tf_target.translation[2] += LINEAR_STEP
                    elif key == 'e': tf_target.translation[2] -= LINEAR_STEP
                    elif key == 'y': tf_target.rotation @= R.from_euler('x', ANGULAR_STEP).as_matrix()
                    elif key == 'i': tf_target.rotation @= R.from_euler('x', -ANGULAR_STEP).as_matrix()
                    elif key == 'u': tf_target.rotation @= R.from_euler('y', ANGULAR_STEP).as_matrix()
                    elif key == 'j': tf_target.rotation @= R.from_euler('y', -ANGULAR_STEP).as_matrix()
                    elif key == 'h': tf_target.rotation @= R.from_euler('z', ANGULAR_STEP).as_matrix()
                    elif key == 'k': tf_target.rotation @= R.from_euler('z', -ANGULAR_STEP).as_matrix()
                    elif key == 'x':
                        print("Exiting...")
                        break
                
                # 持续调用IK求解器
                ik.solve_ik(tf_target.homogeneous)
                time.sleep(1.0 / LOOP_RATE)
        finally:
            # 确保在退出时恢复终端设置
            del kb

    # --- 执行主测试函数 ---
    main_interactive_test()
