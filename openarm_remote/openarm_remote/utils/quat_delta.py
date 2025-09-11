import numpy as np
from scipy.spatial.transform import Rotation as R

class BaseYawCorrector:#关于四元数求出旋转差，修补旋转矩阵的类，专为yaw_delta使用
    def __init__(self, decimal_precision=6):
        """
        decimal_precision: 输出矩阵四舍五入的小数位
        """
        self.decimal_precision = decimal_precision

    @staticmethod
    def rotation_matrix_from_yaw(yaw_rad):
        """生成绕 Z 轴旋转的 3x3 矩阵"""
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
    @staticmethod
    def rotation_matrix_from_pitch(pitch_rad):
        """生成绕 Y 轴旋转的 3x3 矩阵"""
        c, s = np.cos(pitch_rad), np.sin(pitch_rad)
        return np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ])
    def apply_yaw(self, rotation_matrices, yaw, degrees=False):
        """
        rotation_matrices: n x 9
        yaw: 单个角度/弧度 或 长度为 n 的数组
        degrees: 是否为角度
        返回 n x 9 修正后的旋转矩阵
        """
        rotation_matrices = np.asarray(rotation_matrices)
        n = rotation_matrices.shape[0]

        # 如果是单个 yaw，则扩展成 n 个
        if np.isscalar(yaw):
            yaw = np.full(n, yaw)
        else:
            yaw = np.asarray(yaw)
            assert yaw.shape[0] == n, "yaw 数量需与矩阵行数一致"

        if degrees:
            yaw = np.deg2rad(yaw)  # 转为弧度

        corrected = []
        for i in range(n):
            R_orig = rotation_matrices[i].reshape(3,3)
            # original_z = R_orig[:, 2].copy()  # 保存原始Z轴

            R_yaw = self.rotation_matrix_from_yaw(yaw[i])
            R_new = R_yaw @ R_orig  # 左乘，Base 坐标系旋转

            # 强制保持原始Z轴的精确值和方向
            # R_new[:, 2] = original_z
            
            # 确保矩阵仍然是正交的=

            R_new = np.round(R_new, self.decimal_precision)  # 精度处理
            corrected.append(R_new.flatten())

        return np.array(corrected)
    def apply_pitch_correction(self, rotation_matrices, pitch, degrees=False):
        """
        修正旋转矩阵，绕 Y 轴旋转。
        参数:
            rotation_matrices: n x 9 旋转矩阵
            pitch: 单个角度/弧度 或 长度为 n 的数组
            degrees: 是否为角度
        返回:
            修正后的旋转矩阵
        """
        rotation_matrices = np.asarray(rotation_matrices)
        n = rotation_matrices.shape[0]

        if np.isscalar(pitch):
            pitch = np.full(n, pitch)
        else:
            pitch = np.asarray(pitch)
            assert pitch.shape[0] == n, "pitch 数量需与矩阵行数一致"

        if degrees:
            pitch = np.deg2rad(pitch)  # 转为弧度

        corrected = []
        for i in range(n):
            R_orig = rotation_matrices[i].reshape(3, 3)
            R_pitch = self.rotation_matrix_from_pitch(pitch[i])
            R_new = R_pitch @ R_orig  # 左乘，Base 坐标系旋转
            R_new = np.round(R_new, self.decimal_precision)  # 精度处理
            corrected.append(R_new.flatten())

        return np.array(corrected)
    
    def compute_yaw_diff(self,quat_from, quat_to):
        """
        计算从 quat_from 到 quat_to 绕 Z 轴的 YAW 差值

        参数:
            quat_from: 记录的四元数 [ x, y, w,z]
            quat_to:   实际的四元数 [ x, y, z,w]

        返回:
            yaw_diff: 绕 Z 轴旋转的弧度
        """
        # 注意 scipy 的 from_quat 需要 [x, y, z, w] 顺序
        r_from = R.from_quat(quat_from)
        r_to   = R.from_quat(quat_to)

        # 转换为欧拉角 (roll, pitch, yaw)
        euler_from = r_from.as_euler('xyz', degrees=False)
        euler_to   = r_to.as_euler('xyz', degrees=False)

        yaw_diff = euler_to[2] - euler_from[2]  # z轴旋转差值

        # 可选：将角度限制在 [-pi, pi] 范围
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi

        return yaw_diff
    
    # def apply_translation_correction(self, translation, yaw_diff):
    #     """
    #     修正平移量根据yaw_diff旋转
    #     参数:
    #         translation: 原始平移 [x, y, z]
    #         yaw_diff: 绕 Z 轴的旋转差值（弧度）
    #     返回:
    #         修正后的平移
    #     """
    #     # 创建旋转矩阵
    #     R_yaw = self.rotation_matrix_from_yaw(-yaw_diff)
    #     # 对平移进行旋转
    #     corrected_translation = R_yaw @ translation
    #     return corrected_translation
    def apply_translation_correction(self, translation, yaw_diff):
        """
        修正平移量根据yaw_diff旋转
        参数:
            translation: 原始平移 [x, y, z] 或 n×3 的数组
            yaw_diff: 绕 Z 轴的旋转差值（弧度），可以是标量或与translation行数相同的数组
        返回:
            修正后的平移
        """
        translation = np.atleast_2d(translation)
        n = translation.shape[0]
        
        if np.isscalar(yaw_diff):
            yaw_diffs = np.full(n, -yaw_diff)
        else:
            yaw_diffs = -np.asarray(yaw_diff)
            if len(yaw_diffs) != n:
                raise ValueError("yaw_diff的长度必须与translation的行数相同")
        
        corrected_translation = np.empty_like(translation)
        
        for i in range(n):
            R_yaw = self.rotation_matrix_from_yaw(yaw_diffs[i])
            corrected_translation[i] = R_yaw @ translation[i]
        
        if corrected_translation.shape[0] == 1:
            return corrected_translation[0]
        return corrected_translation
    
# ------------------ 示例 ------------------
rotation_matrices = np.array([
    [1,0,0, 0,1,0, 0,0,1],
    [0,-1,0, 1,0,0, 0,0,1]
])
a = [0, 0, 0, 1]  # 单位四元数 [x, y, z, w]
b = [0, 0, 0.7071, 0.7071]  # 绕Z轴90度 [x, y, z, w]
corrector = BaseYawCorrector()
yaw_diff = corrector.compute_yaw_diff(a, b)  # 角度

corrected_matrices = corrector.apply_yaw(rotation_matrices, yaw_diff)

translation = np.array([[1.0, -2.0, 10.0],
                       [1.0, -2.0, 10.0]])

# 根据yaw_diff修正平移
corrected_translation = corrector.apply_translation_correction(translation, yaw_diff)

print("修正后的旋转矩阵：", corrected_matrices)
print("修正后的平移：", corrected_translation)
