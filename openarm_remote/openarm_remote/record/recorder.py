import avro.schema
from avro.datafile import DataFileWriter
from avro.io import DatumWriter
import os
from glob import glob
import pathlib

# SCHEMA_FILE = (pathlib.Path(__file__).parent / 'record.avsc').resolve()

import re
import numpy as np
import json
from ament_index_python.packages import get_package_share_directory

PACKAGE_NAME = "openarm_remote"
try:

    package_share_dir = get_package_share_directory(PACKAGE_NAME)
    SCHEMA_FILE = os.path.join(package_share_dir, "record", "record.avsc")  
except FileNotFoundError:

    SCHEMA_FILE = (pathlib.Path(__file__).parent / "record.avsc").resolve()  

# 解析Schema前先验证文件存在
if not os.path.exists(SCHEMA_FILE):
    raise FileNotFoundError(f"record.avsc not found at: {SCHEMA_FILE}\n请检查路径配置是否正确")

SCHEMA = avro.schema.parse(open(SCHEMA_FILE, "rb").read())

class Recorder:
    def __init__(self, record_dir):
        self.record_dir = record_dir
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)
        self.recording = False
        self.writer = None
        files = glob(f"{self.record_dir}/record_*.avro")
        self.ids  = 0
        pattern = re.compile(r'record_(\d+)\.avro')
        for file in files:
            match = pattern.search(file)
            if match:
                file_id = int(match.group(1))
                if file_id >= self.ids:
                    self.ids = file_id + 1
        print(f"RecorderDetect initialized. Next ID: {self.ids}")

    def start(self):
        if self.recording:
            self.writer.close()
        self.recording = True
        record_file = f"{self.record_dir}/record_{self.ids}.avro"
        self.writer = DataFileWriter(open(record_file, "wb"), DatumWriter(), SCHEMA)
        print(f"Recording started: {record_file}")
    
    def record(self, data: dict):
        if not self.recording or not self.writer:
            print("Recorder is not active. Call start() first.")
            return
        # if "meta" not in data:
        #     data["meta"] = "{}"
        # else:
        #     if isinstance(data["meta"], dict):
        #         data["meta"] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data["meta"].items()}
        #         data["meta"] = json.dumps(data["meta"])
        data_parsed = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}
        self.writer.append(data_parsed)
    
    def stop(self):
        if not self.recording:
            return
        self.recording = False
        if self.writer:
            self.writer.close()
            self.ids += 1
            print("Recording stopped.")
            self.writer = None