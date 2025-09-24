import numpy as np
from glob import glob

# 获取所有的 replay 文件路径
REPLAY_FILES = sorted(glob('/home/usyd/openarm_ws/detect_record/left_arm/simple_*'))
REPLAY_FILES_R = sorted(glob('/home/usyd/openarm_ws/detect_record/right_arm/simple_*'))
# REPLAY_FILES = sorted(glob('/home/usyd/tube_ws/drecorder/simple_*'))
def read_replay_file(file_path):
    """
    读取单个 .npz 文件并打印其内容
    """
    try:
        data = np.load(file_path)
        print(f"Loaded {file_path}")
        
        # 打印文件里的所有键
        print(f"Keys in the file: {data.files}")
        
        # 打印一些数据内容
        for key in data.files:
            print(f"{key}: {data[key]}")
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

def read_all_replay_files():
    """
    读取所有 replay 文件
    """
    for file in REPLAY_FILES:
        read_replay_file(file)
    for file in REPLAY_FILES_R:
        read_replay_file(file)

# 调用函数读取所有文件内容
read_all_replay_files()
