import rclpy
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py.point_cloud2 import read_points
import pandas as pd
import numpy as np
import open3d as o3d
from tqdm import tqdm
import yaml
import os
from pathlib import Path
from datetime import datetime

from .data_io import LidarExtrinsics

def to_datetime(timestamp_ns):
    return datetime.fromtimestamp(timestamp_ns / 1e9)

def get_messages_counts(bag_path):
    yaml_path = os.path.join(bag_path, "metadata.yaml")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    counts = dict()
    for topic_metadata in config['rosbag2_bagfile_information']['topics_with_message_count']:
        topic_name = topic_metadata['topic_metadata']['name']
        counts[topic_name] = topic_metadata['message_count']
    return counts

class PointCloudParserRos:
    def __init__(self, data, field_names=["x", "y", "z", 'intensity'], pc_o3d=True):
        self.data = data
        self.PointCloud2 = get_message('sensor_msgs/msg/PointCloud2')
        self.field_names = field_names
        self.pc_o3d = pc_o3d
    
    def __call__(self):
        msg = deserialize_message(self.data, self.PointCloud2)
        points = np.asarray(read_points(msg, field_names=self.field_names, skip_nans=True))
        if self.pc_o3d:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(np.stack([points[n] for n in ['x','y','z'] if n in self.field_names], axis=-1))
            if 'intensity' in self.field_names:
                pc.colors = o3d.utility.Vector3dVector(np.stack([(points['intensity'].astype(float) / 255.)]*3, axis=-1))
            return pc
        else:
            return pd.DataFrame(points)
    
    def __asarray__(self):
        return self()

class PointCloudLoaderRos:
    def __init__(self, bag_path: Path, topic_name, prescan=False, extrinsics_file=None, pc_o3d=True, verbose=False, interval=1):
        rclpy.init()
        
        # Setup reader
        if not isinstance(bag_path, Path):
            bag_path = Path(bag_path)
        assert bag_path.is_dir(), 'please provide a directory containing *.db3 and metadata.yaml'
        storage_options = StorageOptions(uri=str(bag_path), storage_id='sqlite3')
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

        self.reader = SequentialReader()
        self.reader.open(storage_options, converter_options)

        topic_type_map = {topic.name: topic.type for topic in self.reader.get_all_topics_and_types()}
        print("Topics in the bag:", sorted(topic_type_map.keys()))
        assert topic_name in topic_type_map, f"Topic '{topic_name}' not found in the bag!"
        self.topic_name = topic_name
        self.verbose = verbose
        self.pc_o3d = pc_o3d
        self.extrinsics = LidarExtrinsics(extrinsics_file) if extrinsics_file is not None else None
        self.interval = interval

        self.data = None
        if prescan:
            self.data = list(iter(self)) # store all data
            self.num = len(self.data)
        else:
            self.last_idx = -1
            self.num = get_messages_counts(bag_path)[topic_name] // interval # read counts from metadata.yaml -> its faster than iterating over all data

    def __iter__(self):
        if self.data is None:
            progressbar = tqdm(disable=not self.verbose)
            self.reader.seek(0)
            i = 0
            while self.reader.has_next():
                topic, data, timestamp = self.reader.read_next()
                timestamp = to_datetime(timestamp)
                if topic == self.topic_name:
                    pcd = PointCloudParserRos(data, pc_o3d=self.pc_o3d)() # parse data right away for now
                    if self.extrinsics is not None:
                        T = self.extrinsics[topic.replace('/','')]
                        if self.pc_o3d:
                            pcd = pcd.transform(T)
                        else:
                            raise Exception('not implemented for df')
                    if i % self.interval == 0:
                        yield timestamp, pcd
                    i += 1
                progressbar.update(1)
        else:
            yield from tqdm(self.data, disable=not self.verbose)

    def __getitem__(self, idx):
        if self.data is None:
            if idx == 0: # reset iterator
                self.data_iter = iter(self)
                self.last_idx = -1
            assert self.last_idx+1 == idx, 'enable prescan if you want to access arbitrary indices'
            self.last_idx += 1
            return next(self.data_iter)
        else:
            return self.data[idx]
    
    def __len__(self):
        return self.num




if __name__ == '__main__':
    # Example usage:
    pc_data = PointCloudLoaderRos("/home/thomas/Downloads/rosbags/with_reflectivity/with_reflectivity_0.db3", topic_name='/ivu2', prescan=True, verbose=True)
    print(pc_data[6])
    print(len(pc_data))