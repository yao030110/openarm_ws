import numpy as np
from scipy.spatial.transform import Rotation as R
class BaseRotationCorrector:
    def __init__(self, decimal_precision=6):
        """
        通用旋转修正器：支持绕 X、Y、Z 轴对旋转矩阵进行左乘修正。
        所有操作均在 **世界坐标系** 下执行（左乘）。

        :param decimal_precision: 输出矩阵四舍五入的小数位数
        """
        self.decimal_precision = decimal_precision

    @staticmethod
    def rotation_matrix_from_roll(roll_rad):
        """生成绕 X 轴（滚转）的 3x3 旋转矩阵"""
        c, s = np.cos(roll_rad), np.sin(roll_rad)
        return np.array([
            [1,  0,  0],
            [0,  c, -s],
            [0,  s,  c]
        ])

    @staticmethod
    def rotation_matrix_from_pitch(pitch_rad):
        """生成绕 Y 轴（俯仰）的 3x3 旋转矩阵"""
        c, s = np.cos(pitch_rad), np.sin(pitch_rad)
        return np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ])

    @staticmethod
    def rotation_matrix_from_yaw(yaw_rad):
        """生成绕 Z 轴（偏航）的 3x3 旋转矩阵"""
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

    def _apply_axis_correction(self, rotation_matrices, angle, axis_func, degrees=False):
        """
        内部通用方法：对旋转矩阵应用绕指定轴的修正（左乘）
        
        :param rotation_matrices: n x 9 的数组，每个是 3x3 矩阵展平
        :param angle: 单个值或长度为 n 的数组，表示旋转量（弧度或角度）
        :param axis_func: 对应轴的旋转矩阵生成函数 (e.g., rotation_matrix_from_yaw)
        :param degrees: 是否输入为角度，若为 True 则转为弧度
        :return: n x 9 修正后的旋转矩阵
        """
        rotation_matrices = np.asarray(rotation_matrices)
        n = rotation_matrices.shape[0]

        if np.isscalar(angle):
            angles = np.full(n, angle)
        else:
            angles = np.asarray(angle)
            assert angles.shape[0] == n, "angle 数量必须与矩阵行数一致"

        if degrees:
            angles = np.deg2rad(angles)

        corrected = []
        for i in range(n):
            R_orig = rotation_matrices[i].reshape(3, 3)
            R_delta = axis_func(angles[i])
            R_new = R_delta @ R_orig  # 左乘：在世界坐标系下旋转
            R_new = np.round(R_new, self.decimal_precision)  # 精度处理
            corrected.append(R_new.flatten())

        return np.array(corrected)

    def apply_roll_correction(self, rotation_matrices, roll, degrees=False):
        """
        绕 X 轴（滚转）修正旋转矩阵。

        :param rotation_matrices: n x 9 的数组，每个是 3x3 矩阵展平
        :param roll: 单个值或数组，表示绕 X 轴的旋转量（默认弧度）
        :param degrees: 若为 True，则输入为角度
        :return: n x 9 修正后的旋转矩阵
        """
        return self._apply_axis_correction(rotation_matrices, roll, self.rotation_matrix_from_roll, degrees)

    def apply_pitch_correction(self, rotation_matrices, pitch, degrees=False):
        """
        绕 Y 轴（俯仰）修正旋转矩阵。

        :param rotation_matrices: n x 9 的数组，每个是 3x3 矩阵展平
        :param pitch: 单个值或数组，表示绕 Y 轴的旋转量（默认弧度）
        :param degrees: 若为 True，则输入为角度
        :return: n x 9 修正后的旋转矩阵
        """
        return self._apply_axis_correction(rotation_matrices, pitch, self.rotation_matrix_from_pitch, degrees)

    def apply_yaw_correction(self, rotation_matrices, yaw, degrees=False):
        """
        绕 Z 轴（偏航）修正旋转矩阵。

        :param rotation_matrices: n x 9 的数组，每个是 3x3 矩阵展平
        :param yaw: 单个值或数组，表示绕 Z 轴的旋转量（默认弧度）
        :param degrees: 若为 True，则输入为角度
        :return: n x 9 修正后的旋转矩阵
        """
        return self._apply_axis_correction(rotation_matrices, yaw, self.rotation_matrix_from_yaw, degrees)

    def compute_rotation_diff(self, quat_from, quat_to, axis='z', degrees=False):
        """
        计算从 quat_from 到 quat_to 在指定轴上的旋转差值（仅提取该轴分量）

        :param quat_from: 原始四元数 [x, y, z, w]
        :param quat_to: 目标四元数 [x, y, z, w]
        :param axis: 'x' / 'y' / 'z' —— 指定要提取的旋转轴
        :param degrees: 是否返回角度而非弧度
        :return: 单个角度值（弧度或角度）
        """
        r_from = R.from_quat(quat_from)
        r_to   = R.from_quat(quat_to)

        euler_from = r_from.as_euler('xyz', degrees=False)
        euler_to   = r_to.as_euler('xyz', degrees=False)

        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if axis not in axis_map:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        diff = euler_to[axis_map[axis]] - euler_from[axis_map[axis]]
        # 归一化到 [-π, π]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        if degrees:
            diff = np.rad2deg(diff)

        return diff

    def apply_translation_correction(self, translation, angle, axis='z', degrees=False):
        """
        根据指定轴的旋转差值，修正平移向量（反向旋转）

        :param translation: n×3 或 3 维平移向量
        :param angle: 旋转差值（弧度或角度）
        :param axis: 'x', 'y', 'z'
        :param degrees: 输入是否为角度
        :return: 修正后的平移向量
        """
        translation = np.atleast_2d(translation)
        n = translation.shape[0]

        if np.isscalar(angle):
            angles = np.full(n, angle)
        else:
            angles = np.asarray(angle)
            if len(angles) != n:
                raise ValueError("angle 长度必须与 translation 行数一致")

        if degrees:
            angles = np.deg2rad(angles)

        axis_map = {'x': self.rotation_matrix_from_roll,
                    'y': self.rotation_matrix_from_pitch,
                    'z': self.rotation_matrix_from_yaw}

        if axis not in axis_map:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        corrected = np.empty_like(translation)
        for i in range(n):
            R_delta = axis_map[axis](-angles[i])  # 反向旋转以补偿
            corrected[i] = R_delta @ translation[i]

        if corrected.shape[0] == 1:
            return corrected[0]
        return corrected
    
# ------------------ 示例 ------------------
if __name__ == "__main__":
    rotation_matrices = np.array([
        [1,0,0, 0,1,0, 0,0,1],
        [0,-1,0, 1,0,0, 0,0,1]
    ])
    a = [0, 0, 0, 1]  # 单位四元数 [x, y, z, w]
    b = [0, 0, 0.7071, 0.7071]  # 绕Z轴90度 [x, y, z, w]
    corrector = BaseRotationCorrector(decimal_precision=6)
    yaw_diff = corrector.compute_rotation_diff(a, b , axis='z', degrees=True)  # 角度

    corrected_matrices = corrector.apply_yaw_correction(rotation_matrices, yaw_diff, degrees=True)

    translation = np.array([[1.0, -2.0, 10.0],
                        [1.0, -2.0, 10.0]])

    # 根据yaw_diff修正平移
    corrected_translation = corrector.apply_translation_correction(translation, yaw_diff,degrees=True)

    print("修正后的旋转矩阵：", corrected_matrices)
    print("修正后的平移：", corrected_translation)