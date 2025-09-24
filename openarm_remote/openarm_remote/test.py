from avro.datafile import DataFileReader
from avro.io import DatumReader
import numpy as np
from glob import glob
from rdp import rdp
import re
from tqdm import tqdm
from pathlib import Path
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

def simple_traj():
        obs = {
            'timestamp': time.time(),
            'ee_pose': np.array([1,2,3]),
            'ee_rot': np.array([1,2,3]),
            'position': np.array([1,2,3]),
            'action':  np.array([1.0,2,3]),
            }
        obs['action'][2] = 10
        print(obs)

if __name__ == "__main__":
    simple_traj()