from avro.datafile import DataFileReader
from avro.io import DatumReader
import numpy as np
from glob import glob
from rdp import rdp
import re
from tqdm import tqdm
from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
import os
package_share_directory = get_package_share_directory('openarm_remote')
config_path = os.path.join(package_share_directory, 'config', 'record_path.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
ws_path = config['files_path']['ws_path']
files = glob(os.path.join(ws_path, "recordings", "record_*.avro"))
# files = glob("/home/usyd/tube_ws/drecorder/record_*.avro")
pattern = re.compile(r"record_(\d+).avro")
def find_hand_action_changes(action):
    """
    返回布尔掩码，标记所有为1、-1，以及从1/-1跳变到0的点。
    """
    action = np.asarray(action).flatten()  # 保证是一维
    mask = np.zeros(len(action), dtype=bool)# 初始化全为 False 的布尔数组
    # 标记所有为1或-1的点
    # mask[(action == 1) | (action == -1)] = True
    # mask[action != 0] = True  # 尝试
    # 标记从1/-1跳变到0的点
    for i in range(1, len(action)):
        # if (action[i-1] in (1, -1)) and action[i] == 0:
        if (action[i-1] !=0) and action[i] == 0:
            mask[i] = True
        elif (action[i-1] == 0) and (action[i] in (1, -1)):
            mask[i-1:i+1] = True
    return mask

def simple_traj(file: str):
    try:
        ep_id = int(pattern.search(file).group(1))
        simple_traj_file = f"simple_{ep_id}.npz"
        # simple_traj_file = f"/home/usyd/tube_ws/drecorder/simple_{ep_id}.npz"
        print(f"Processing episode {ep_id} from file {file}")
        
        # 读取Avro文件
        reader = DataFileReader(open(file, "rb"), DatumReader())
        ee_pose = []
        ee_rot = []
        action = []
        for sample in reader:
            ee_pose.append(sample['ee_pose'])
            ee_rot.append(sample['ee_rot'])
            action.append(sample['action'][:][6])
            # action.append(sample['action'][:])
        
        
        
        # 转换为numpy数组
        ee_pose = np.array(ee_pose)
        ee_rot = np.array(ee_rot)
        action = np.array(action)
        # knuckle_pose = np.array(sample['knuckle_pose'])
        # traj_file = Path("/home/usyd/aubo_ws/detect_record")
        traj_file = Path(ws_path) / "detect_record"
        simple_traj_file = traj_file / simple_traj_file
        output_dir = simple_traj_file.parent
      # 2. 创建目录，如果它不存在的话
      #    exist_ok=True 是一个安全开关，如果文件夹已经存在，则不会报错
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取手部动作变化的关键点
        hand_action = action.copy()  # 复制action数组，避免直接修改原始数据
        hand_action_changes = find_hand_action_changes(hand_action)
        # # 将手部动作差分
        # hand_action_diff = action.copy()
        # hand_action_diff[1:] -= hand_action_diff[:-1]
        
        # 使用RDP简化ee_pose
        epsilon = 0.001
        ee_pose_mask = rdp(ee_pose, epsilon=epsilon, return_mask=True)
        
        # 仅保留变化点和结束的0
        # hand_action_vaild = np.zeros(hand_action.shape[0], dtype=bool)
        
        # hand_action_vaild = hand_action_diff[:, -6:] != 0
        # hand_action_vaild = hand_action_vaild.any(axis=1)
        #返回一个布尔数组，标记所需的帧
        
        ee_pose_mask = ee_pose_mask | hand_action_changes
        #
        np.savez_compressed(simple_traj_file, ee_pose=ee_pose[ee_pose_mask], ee_rot=ee_rot[ee_pose_mask], action=action[ee_pose_mask])
        print(f"Saved simplified trajectory to {simple_traj_file}")
    except Exception as e:
        print(f"Error processing file {file}: {e}")

if __name__ == "__main__":
    for file in tqdm(files):
        simple_traj(file)
    print("All files processed.")