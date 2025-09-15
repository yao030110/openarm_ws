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
    def __init__(self, Visualization=False, filter=False):
        # __file__ 是当前脚本(mod_ik.py)的路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        package_root_dir = os.path.dirname(script_dir)
        config_path = os.path.join(package_root_dir, 'config', 'robot_control.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.filter = filter
        self.Visualization = Visualization
        self.robot_ik = config['robot_for_ik']
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
    from scipy.spatial.transform import Rotation as R
    ik = General_ArmIK(Visualization=True)
    rotation_speed = 0.005
    noise_amplitude_translation = 0.001
    noise_amplitude_rotation = 0.01
    R_down = pin.utils.rotate('x', np.pi)  # 获取绕X轴180度的旋转矩阵
    q_down = pin.Quaternion(R_down)  
    tf_target = pin.SE3(
        q_down,
        np.array([0.20, 0.20, 0.1]),
    )
    # 一个向下的姿态
    user_input = input(
        "Please enter the start signal (enter 's' to start the subsequent program):\n")
    if user_input.lower() == 's':
        step = 0
        while True:
            # Apply rotation noise with bias towards y and z axes
            rotation_noise = pin.Quaternion(
                np.cos(np.random.normal(0, noise_amplitude_rotation) / 2), 0, np.random.normal(0, noise_amplitude_rotation / 2), 0).normalized()  # y bias
            if step <= 120:
                angle = rotation_speed * step
                # 在“Z向下”的基础上，绕 Y 轴旋转（即左右摆动）
                R_y = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0).toRotationMatrix()
                tf_target.rotation = (rotation_noise * pin.Quaternion(R_y)).toRotationMatrix()
                tf_target.translation += (np.array([0.001, 0.001, 0.001]) + np.random.normal(0, noise_amplitude_translation, 3))

            else:
                angle = rotation_speed * (240 - step)
                R_y = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0).toRotationMatrix()
                tf_target.rotation = (rotation_noise * pin.Quaternion(R_y)).toRotationMatrix()
                tf_target.translation -= (np.array([0.001, 0.001, 0.001]) + np.random.normal(0, noise_amplitude_translation, 3))

            ik.solve_ik(tf_target.homogeneous)

            step += 1
            if step > 240:
                step = 0
            time.sleep(0.1)
