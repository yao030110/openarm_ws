import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import Joy
class TrajectoryGenerator:
    """
    一个可重用的类，用于根据关键帧生成平滑的机器人末端执行器轨迹。
    它对位置和旋转进行插值，并处理手部动作的变化和关键帧的等待。
    """
    def __init__(self, fps=30, hand_wait_time=1.0, keyframe_wait_time=0.0, eps_min=1e-3, eps_max=5e-3, acceleration=5e-4):
        """
        初始化轨迹生成器。

        参数
        ----
        fps                : int     轨迹的采样率 (帧/秒)
        hand_wait_time     : float   动作发生变化后，需要保持新动作的时间 (秒)
        keyframe_wait_time : float   在每个移动关键帧后等待的时间 (秒)。
        eps_min/max        : float   单步线性距离（可视为速度）的最小/最大值。
        acceleration       : float   每一帧的速度变化量（米/帧²），一个固定的加速度值。
        """
        self.fps = fps
        self.hand_wait_time = hand_wait_time
        self.keyframe_wait_time = keyframe_wait_time
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.acceleration = acceleration
        
        self.hand_wait_frames = int(round(self.hand_wait_time * self.fps))
        self.keyframe_wait_frames = int(round(self.keyframe_wait_time * self.fps))
    
    def _interp_segment(self, p0, p1, r0, r1, a_copy,eps_min,eps_max):
        """
        私有方法：在两个关键帧之间插值生成一个轨迹段。
        (此方法与上一版本相同，无需修改)
        """
        # --- 1. 位置插值 ---
        distance = np.linalg.norm(p1 - p0)#位置帧的距离，计算向量长度的，这里是直接数值
        if distance < 1e-9:
            return (np.vstack([p0, p1]),
                    np.vstack([r0, r1]),
                    np.vstack([a_copy, a_copy]))

        avg_eps = (eps_min + eps_max) / 2.0
        num_points = int(np.ceil(distance / avg_eps))#步数
        if num_points < 2:
            num_points = 2

        # --- 基于恒定加速度生成速度剖面 ---
        delta_v = eps_max - eps_min
        if self.acceleration <= 1e-9:
             num_ramp_points = 0
        else:
            num_ramp_points = int(np.ceil(delta_v / self.acceleration))#大小速度差值与加速的的商，规定几步达到最大速度

        if num_points >= 2 * num_ramp_points and num_ramp_points > 0:
            # 梯形剖面
            num_const_points = num_points - 2 * num_ramp_points #梯形顶端的步数
            #在指定的区间内，生成指定数量的、等间距的数值（一维数组）np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
            ramp_up = np.linspace(eps_min, eps_max, num_ramp_points)
            const_phase = np.full(num_const_points, eps_max)#填充num_const_points大小的数组值
            ramp_down = np.linspace(eps_max, eps_min, num_ramp_points)
            step_sizes = np.concatenate((ramp_up, const_phase, ramp_down))#拼接
        else:
            # 三角形剖面
            num_accel_points = num_points // 2#整数除法，向下取整，一个/是浮点型除法
            num_decel_points = num_points - num_accel_points
            peak_eps = min(eps_min + self.acceleration * num_accel_points, eps_max)
            ramp_up = np.linspace(eps_min, peak_eps, num_accel_points)
            ramp_down = np.linspace(peak_eps, eps_min, num_decel_points)
            step_sizes = np.concatenate((ramp_up, ramp_down))

        if len(step_sizes) != num_points:
             step_sizes = np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, len(step_sizes)), step_sizes)

        s = np.cumsum(step_sizes[:-1])
        s = np.insert(s, 0, 0.0)
        if s[-1] > 1e-9:
            s /= s[-1]
        
        interpolated_poses = p0[None, :] + (p1 - p0)[None, :] * s[:, None]
        interpolated_actions = np.repeat(a_copy[None, :], len(interpolated_poses), axis=0)

        # --- 2. 旋转插值 (Slerp) ---
        rot_start = Rotation.from_matrix(r0.reshape(3, 3))
        rot_end = Rotation.from_matrix(r1.reshape(3, 3))
        key_rotations = Rotation.concatenate([rot_start, rot_end])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rotations)
        # 让旋转在位置完成前就结束（比如在70%位置处完成100%旋转）
        rotation_completion_ratio = 0.7  # 在70%的位置路径上完成100%旋转
        rotation_s = np.clip(s / rotation_completion_ratio, 0, 1)

        interpolated_rotations_obj = slerp(rotation_s)        
        # interpolated_rotations_obj = slerp(s)
        interpolated_rotations = interpolated_rotations_obj.as_matrix().reshape(-1, 9)

        return interpolated_poses, interpolated_rotations, interpolated_actions

    def generate_Z5cm(self, ee_pose, ee_rot, hand_action):
        """
        根据给定的关键帧生成轨迹。
        """
        ee_pose = np.asarray(ee_pose, dtype=float)
        ee_rot = np.asarray(ee_rot, dtype=float)
        hand_action = np.asarray(hand_action, dtype=int)

        assert ee_pose.ndim == 2 and ee_pose.shape[1] == 3, "ee_pose 必须是 (N,3) 形状"
        assert ee_rot.ndim == 2 and ee_rot.shape[1] == 9, "ee_rot 必须是 (N,9) 形状"
        assert len(ee_pose) == len(ee_rot) == len(hand_action), "所有输入的关键帧数量必须一致"

        new_p, new_r, new_a = [], [], []

        def append_frame(p, r, a):
            new_p.append(p)
            new_r.append(r)
            new_a.append(a)

        z_min = np.min(ee_pose[:, 2])
        
        for i in range(len(ee_pose)):
            if i == 0:
                append_frame(ee_pose[0], ee_rot[0], hand_action[0])
                # 在第一个点之后也应用等待
                if self.keyframe_wait_frames > 0:
                   for _ in range(self.keyframe_wait_frames):
                       append_frame(new_p[-1], new_r[-1], new_a[-1])
                continue

            p_prev, r_prev, a_prev = ee_pose[i-1], ee_rot[i-1], hand_action[i-1]
            p_curr, r_curr, a_curr = ee_pose[i], ee_rot[i], hand_action[i]
            p_start, r_start, a_interp = np.asarray(new_p[-1]), np.asarray(new_r[-1]), np.asarray(new_a[-1])
            
            if (p_curr[2] - z_min) <= 0.04:
                interp_p, interp_r, interp_a = self._interp_segment(p_start, p_curr, r_start, r_curr, a_interp,2e-3,2e-3)
            else :
                interp_p, interp_r, interp_a = self._interp_segment(p_start, p_curr, r_start, r_curr, a_interp,self.eps_min, self.eps_max)

            if len(interp_p) > 1:
                append_slice = slice(1, None)
                new_p.extend(interp_p[append_slice])
                new_r.extend(interp_r[append_slice])
                new_a.extend(interp_a[append_slice])
            
            # 1. 更新刚刚到达的关键帧的最终手部动作
            new_a[-1] = a_curr.copy()

            # 2. 如果手部动作发生变化，应用手部动作的专属等待
            if not np.array_equal(a_curr, a_prev):
                if self.hand_wait_frames > 0:
                    for _ in range(self.hand_wait_frames):
                        append_frame(new_p[-1], new_r[-1], new_a[-1])

            # 3. 应用通用的移动关键帧等待（在手部等待之后）
            if self.keyframe_wait_frames > 0:
                for _ in range(self.keyframe_wait_frames):
                    append_frame(new_p[-1], new_r[-1], new_a[-1])
            
            # <--- MODIFICATION END --->

        return np.asarray(new_p), np.asarray(new_r), np.asarray(new_a)
    def generate_slow(self, ee_pose, ee_rot, hand_action):
        """
        根据给定的关键帧生成轨迹。
        """
        ee_pose = np.asarray(ee_pose, dtype=float)
        ee_rot = np.asarray(ee_rot, dtype=float)
        hand_action = np.asarray(hand_action, dtype=int)

        assert ee_pose.ndim == 2 and ee_pose.shape[1] == 3, "ee_pose 必须是 (N,3) 形状"
        assert ee_rot.ndim == 2 and ee_rot.shape[1] == 9, "ee_rot 必须是 (N,9) 形状"
        assert len(ee_pose) == len(ee_rot) == len(hand_action), "所有输入的关键帧数量必须一致"

        new_p, new_r, new_a = [], [], []

        def append_frame(p, r, a):
            new_p.append(p)
            new_r.append(r)
            new_a.append(a)

        
        for i in range(len(ee_pose)):
            if i == 0:
                append_frame(ee_pose[0], ee_rot[0], hand_action[0])
                # 在第一个点之后也应用等待
                if self.keyframe_wait_frames > 0:
                   for _ in range(self.keyframe_wait_frames):
                       append_frame(new_p[-1], new_r[-1], new_a[-1])
                continue

            p_prev, r_prev, a_prev = ee_pose[i-1], ee_rot[i-1], hand_action[i-1]
            p_curr, r_curr, a_curr = ee_pose[i], ee_rot[i], hand_action[i]
            p_start, r_start, a_interp = np.asarray(new_p[-1]), np.asarray(new_r[-1]), np.asarray(new_a[-1])
            
            interp_p, interp_r, interp_a = self._interp_segment(p_start, p_curr, r_start, r_curr, a_interp,1e-3,3e-3)
            
            if len(interp_p) > 1:
                append_slice = slice(1, None)
                new_p.extend(interp_p[append_slice])
                new_r.extend(interp_r[append_slice])
                new_a.extend(interp_a[append_slice])
            
            # 1. 更新刚刚到达的关键帧的最终手部动作
            new_a[-1] = a_curr.copy()

            # 2. 如果手部动作发生变化，应用手部动作的专属等待
            if not np.array_equal(a_curr, a_prev):
                if self.hand_wait_frames > 0:
                    for _ in range(self.hand_wait_frames):
                        append_frame(new_p[-1], new_r[-1], new_a[-1])

            # 3. 应用通用的移动关键帧等待（在手部等待之后）
            if self.keyframe_wait_frames > 0:
                for _ in range(self.keyframe_wait_frames):
                    append_frame(new_p[-1], new_r[-1], new_a[-1])
            
            # <--- MODIFICATION END --->

        return np.asarray(new_p), np.asarray(new_r), np.asarray(new_a)

    def generate(self, ee_pose, ee_rot, hand_action):
        """
        根据给定的关键帧生成轨迹。
        """
        ee_pose = np.asarray(ee_pose, dtype=float)
        ee_rot = np.asarray(ee_rot, dtype=float)
        hand_action = np.asarray(hand_action, dtype=int)

        assert ee_pose.ndim == 2 and ee_pose.shape[1] == 3, "ee_pose 必须是 (N,3) 形状"
        assert ee_rot.ndim == 2 and ee_rot.shape[1] == 9, "ee_rot 必须是 (N,9) 形状"
        assert len(ee_pose) == len(ee_rot) == len(hand_action), "所有输入的关键帧数量必须一致"

        new_p, new_r, new_a = [], [], []

        def append_frame(p, r, a):
            new_p.append(p)
            new_r.append(r)
            new_a.append(a)

        z_min = np.min(ee_pose[:, 2])
        
        for i in range(len(ee_pose)):
            if i == 0:
                append_frame(ee_pose[0], ee_rot[0], hand_action[0])
                # 在第一个点之后也应用等待
                if self.keyframe_wait_frames > 0:
                   for _ in range(self.keyframe_wait_frames):
                       append_frame(new_p[-1], new_r[-1], new_a[-1])
                continue

            p_prev, r_prev, a_prev = ee_pose[i-1], ee_rot[i-1], hand_action[i-1]
            p_curr, r_curr, a_curr = ee_pose[i], ee_rot[i], hand_action[i]
            p_start, r_start, a_interp = np.asarray(new_p[-1]), np.asarray(new_r[-1]), np.asarray(new_a[-1])
            
            interp_p, interp_r, interp_a = self._interp_segment(p_start, p_curr, r_start, r_curr, a_interp,self.eps_min, self.eps_max)

            if len(interp_p) > 1:
                append_slice = slice(1, None)
                new_p.extend(interp_p[append_slice])
                new_r.extend(interp_r[append_slice])
                new_a.extend(interp_a[append_slice])
            
            # 1. 更新刚刚到达的关键帧的最终手部动作
            new_a[-1] = a_curr.copy()

            # 2. 如果手部动作发生变化，应用手部动作的专属等待
            if not np.array_equal(a_curr, a_prev):
                if self.hand_wait_frames > 0:
                    for _ in range(self.hand_wait_frames):
                        append_frame(new_p[-1], new_r[-1], new_a[-1])

            # 3. 应用通用的移动关键帧等待（在手部等待之后）
            if self.keyframe_wait_frames > 0:
                for _ in range(self.keyframe_wait_frames):
                    append_frame(new_p[-1], new_r[-1], new_a[-1])
            
            # <--- MODIFICATION END --->

        return np.asarray(new_p), np.asarray(new_r), np.asarray(new_a)

def calculate_relative_actions(target_poses, target_rots, hand_poses, euler_convention='xyz'):
    """
    将一系列绝对位姿（位置+旋转矩阵）和手部姿态转换为相对动作指令。

    参数:
    - target_poses (list or np.ndarray): N个目标位置的序列，每个位置是 (x, y, z)。形如 (N, 3)。
    - target_rots (list or np.ndarray): N个目标旋转矩阵的序列，每个是 (3, 3) 的矩阵。形如 (N, 3, 3)。
    - hand_poses (list or np.ndarray): N个手部姿态的序列（例如，夹爪开合），每个是1维向量。形如 (N, 1)。
    - euler_convention (str): 转换欧拉角时使用的轴顺序，例如 'xyz', 'zyx'。这必须与你的机器人控制器或仿真环境相匹配。

    返回:
    - np.ndarray: 一个包含 N-1 个动作的数组。每个动作是一个12维向量：
     [delta_x, delta_y, delta_z, delta_euler_x, delta_euler_y, delta_euler_z, hand_pose]
     """
    # 确保所有输入列表的长度一致
    num_poses = len(target_poses)
    if not (num_poses == len(target_rots) == len(hand_poses)):
        raise ValueError("输入序列的长度必须相同。")

    if num_poses < 2:
        print("警告：位姿序列少于2个，无法计算相对动作。")
        return np.array([])

    # 将输入转换为numpy数组以进行高效计算
    target_poses = np.array(target_poses)
    target_rots = np.array(target_rots)
    hand_poses = np.array(hand_poses)

    actions = []

    # 从第二个位姿开始遍历，以计算与前一个位姿的相对变化
    for i in range(1, num_poses):
        # -- 1. 获取当前和前一个状态 --
        prev_pos = target_poses[i - 1]
        current_pos = target_poses[i]

        prev_rot_mat = target_rots[i - 1]
        current_rot_mat = target_rots[i]

        # 动作指令通常与目标状态相关联，所以我们使用当前帧的hand_pose
        current_hand_pose = hand_poses[i]

        # -- 2. 计算相对位移 --
        delta_pos = current_pos - prev_pos

        # -- 3. 计算相对旋转 --
        # R_rel = R_prev^{-1} * R_current
        # 对于旋转矩阵，其逆矩阵等于其转置
        relative_rot_mat = prev_rot_mat.T @ current_rot_mat

        # 将相对旋转矩阵转换为欧拉角
        # 注意：使用与你的系统匹配的欧拉角顺序！
        # 结果默认是弧度（radians），这在机器人控制中是标准的
        relative_rotation = Rotation.from_matrix(relative_rot_mat)
        delta_euler = relative_rotation.as_euler(euler_convention, degrees=False)

        # -- 4. 组合成最终的12维动作向量 --
        # [delta_pos (3), delta_euler (3), hand_pose (1)]
        action = np.concatenate([delta_pos, delta_euler, current_hand_pose])
        actions.append(action)

    return np.array(actions)    

def simplify_6dof_trajectory_to_mask(points, epsilon_pos, epsilon_rot_deg):
    """
    使用 Ramer-Douglas-Peucker (RDP) 算法简化6DoF轨迹，并返回一个布尔掩码。

    参数:
    - points (list of tuples): 轨迹点列表。每个点是 (position, rotation)。
    - epsilon_pos (float): 位置误差的最大容忍度。
    - epsilon_rot_deg (float): 旋转误差的最大容忍度（度）。

    返回:
    - numpy.ndarray: 与 `points` 等长的布尔数组。True表示保留该点。
    """
    n_points = len(points)
    if n_points < 3:
        return np.full(n_points, True)

    # 1. 初始化掩码，默认只保留起点和终点
    mask = np.full(n_points, False)
    mask[0] = True
    mask[-1] = True

    epsilon_rot_rad = np.deg2rad(epsilon_rot_deg)

    # 2. 创建一个基于索引的递归函数
    # 使用栈 (stack) 代替递归，可以避免Python的递归深度限制，处理超长轨迹更稳健
    stack = [(0, n_points - 1)]

    while stack:
        start_idx, end_idx = stack.pop()
        
        # 如果段内没有中间点，则跳过
        if end_idx <= start_idx + 1:
            continue

        start_point = points[start_idx]
        end_point = points[end_idx]
        start_pos, start_rot = start_point
        end_pos, end_rot = end_point
        
        max_dist = -1.0
        max_idx = -1

        slerp = Slerp([0, 1], Rotation.concatenate([start_rot, end_rot]))
        
        # 遍历此段内的中间点
        for i in range(start_idx + 1, end_idx):
            current_pos, current_rot = points[i]
            
            # 位置误差计算 (与之前相同)
            line_vec = end_pos - start_pos
            point_vec = current_pos - start_pos
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0:
                dist_pos = np.linalg.norm(point_vec)
            else:
                t = np.dot(point_vec, line_vec) / line_len_sq
                if t < 0: dist_pos = np.linalg.norm(current_pos - start_pos)
                elif t > 1: dist_pos = np.linalg.norm(current_pos - end_pos)
                else:
                    projection = start_pos + t * line_vec
                    dist_pos = np.linalg.norm(current_pos - projection)
            
            # 旋转误差计算
            # **注意**: 时间比例现在要根据索引来计算
            time_ratio = (i - start_idx) / (end_idx - start_idx)
            interp_rot = slerp([time_ratio])[0]
            dist_rot = (interp_rot.inv() * current_rot).magnitude()
            
            # 组合误差
            normalized_dist = max(dist_pos / epsilon_pos, dist_rot / epsilon_rot_rad)

            if normalized_dist > max_dist:
                max_dist = normalized_dist
                max_idx = i

        # 如果最大误差超过阈值，则标记该点，并将两边的子段入栈
        if max_dist > 1.0:
            mask[max_idx] = True
            stack.append((start_idx, max_idx))
            stack.append((max_idx, end_idx))
            
    return mask