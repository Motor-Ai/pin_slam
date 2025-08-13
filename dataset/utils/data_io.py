import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import datetime
import os
import yaml, json

def matrix_to_pq(transform, flatten=False):
    quad = R.from_matrix(transform[:3,:3])
    pos = transform[:3,3]
    if flatten:
        return [*pos, *quad]
    return pos, quad

def get_matrix(pos, quad):
    T = np.eye(4)
    T[:3,:3] = R.from_quat(quad, scalar_first=False).as_matrix()
    T[:3,3] = pos
    return T



def string_to_timestamp(timestamp_str, isdir=False):
    if isdir:
        timestamp_str = os.path.basename(timestamp_str)
    # Split into seconds and nanoseconds
    seconds_str, nanoseconds_str = timestamp_str.split("_")
    seconds = int(seconds_str)
    nanoseconds = int(nanoseconds_str)
    # Create datetime from seconds
    dt = datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)
    # Add nanoseconds as microseconds (Python datetime doesn't support nanoseconds natively)
    # We truncate nanoseconds to microseconds
    microseconds = nanoseconds // 1000
    dt = dt.replace(microsecond=microseconds)
    # Display with microsecond precision
    return dt

def dict_to_timestamp(data):
    if 'timestamp' in data:
        sec = data['timestamp']['sec']
        nanosec = data['timestamp']['nanosec']
    elif 'timestamp.sec' in data:
        sec = data['timestamp.sec']
        nanosec = data['timestamp.nanosec']
    else:
        raise Exception()
    return get_timestamp(sec, nanosec)

def get_timestamp(sec, nanosec):
    # Convert seconds to datetime
    dt = datetime.datetime.fromtimestamp(sec, tz=datetime.timezone.utc)
    # Add nanoseconds as microseconds (Python datetime only supports microseconds)
    microsec = nanosec // 1000
    dt = dt + datetime.timedelta(microseconds=microsec)
    return dt

def concatenate_point_clouds(pcs):
        if len(pcs) == 1:
            return pcs[0]
        
        combined = o3d.geometry.PointCloud()
        for pc in pcs:
            combined += pc
        return combined

class LidarExtrinsics:
    def __init__(self, extrinsics_file):
        self.extrinsics = self.load_config(extrinsics_file)

    @staticmethod
    def load_config(yaml_path):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        if 'sensors' in config:
            sensors = config['sensors']
            base_T = np.eye(4)
        else:
            sensors = []
            lidar_pq = np.array(config['transforms_list']).reshape(-1,7)
            lidar_positions = config['lidar_position']
            #lidar_transforms = np.array([get_matrix(transform_data[:3], transform_data[3:]) for transform_data in lidar_pq])
            base_link_transform = np.asarray(config['base_link_transform'])
            base_T = get_matrix(base_link_transform[:3], base_link_transform[3:]) 
            lidar_file_names = config['lidar_file_names']
            for file_name, transform, position in zip(lidar_file_names, lidar_pq, lidar_positions):
                item = dict(
                    translation = transform[:3],
                    rotation_quaternion = transform[3:],
                    file_name=file_name,
                    position=position
                )
                sensors.append(item)
            
        result = dict()
        for sensor in sensors:
            file_name = sensor['file_name']
            t = np.array(sensor['translation'])
            q = np.array(sensor['rotation_quaternion'])
            T = base_T @ get_matrix(t, q)
            result[file_name] = T

        return result
    
    def __getitem__(self, name):
        return self.extrinsics[name]
    

class PointCloudsLoader:
    pre_load = False

    def __init__(self, pc_names, extrinsics_file=None, metadata_filename='reconstruction_metadata.json', dirnames=None):
        self.pc_names = pc_names
        self.extrinsics = LidarExtrinsics(extrinsics_file) if extrinsics_file is not None else None
        self.metadata_filename = metadata_filename
        self.dirnames = dirnames
        
    def __call__(self, dir_path):
        # get individuel timestamps
        with open(os.path.join(dir_path, self.metadata_filename), 'r') as f:
            data = json.load(f)
        timestamps = [dict_to_timestamp(data[pos_name][0]) for pos_name in self.pc_names]

        # load point clouds
        pcds = []
        for pc_name in self.pc_names:
            pcd = o3d.io.read_point_cloud(os.path.join(dir_path, f"{pc_name}.pcd"))
            if self.extrinsics is not None:
                T = self.extrinsics[pc_name]
                pcd = pcd.transform(T)
            pcds.append(pcd)

        #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,0])
        #o3d.visualization.draw_geometries([mesh_frame]+pcds)

        return timestamps, pcds
    
    def __iter__(self, dirnames=None):
        if dirnames is None:
            dirnames = self.dirnames
        assert dirnames is not None
        for dirname in dirnames:
            yield self(dirname)

    def __len__(self):
        assert self.dirnames is not None
        return len(self.dirnames)
    
    def __getitem__(self, idx):
        assert self.dirnames is not None
        return self(self.dirnames[idx])

class IMULoader:
    pre_load = True

    def __init__(self, filename_glob='global_position_history.json', filename_acc='localization_history.json'):
        self.filename_glob = filename_glob
        self.filename_acc = filename_acc

    def load_gps(self, dir_path):
        with open(os.path.join(dir_path, self.filename_glob), 'r') as f:
            data = json.load(f)
        return PoseInterpolation2D(data['global_pos_history'])
    
    def load_acc_twist(self, dir_path, timestamps):
        with open(os.path.join(dir_path, self.filename_acc), 'r') as f:
            data = json.load(f)
        acc = AccInterpolation(timestamps, data['accels'])
        twist = TwistInterpolation(timestamps, data['twists'])
        return acc, twist
    
    def __call__(self, dir_path):
        gps = self.load_gps(dir_path)
        acc, twist = self.load_acc_twist(dir_path, gps.timestamps)
        return gps, acc, twist
    
class PoseInterpolation2D:
    # x,y in meters.
    # yaw in rad.

    def __init__(self, x, *args):
        if len(args) == 0:
            self.timestamps = np.array([dict_to_timestamp(d) for d in x])
            self.xy = np.array([(d['local_xy_yaw']['x'], d['local_xy_yaw']['y']) for d in x])
            self.yaw = np.array([d['local_xy_yaw']['yaw'] for d in x])
        else:
            self.timestamps, self.xy, self.yaw = (x, *args)

    def interpolate(self, time):
        t0 = self.timestamps[0]
        timestamps = np.array([(t-t0).total_seconds() for t in self.timestamps])
        time = ((time-t0)).total_seconds()

        if time <= timestamps[0]:
            print('Warning:', timestamps[0], timestamps[-1], time)
            return self.xy[0], self.yaw[0]
        if time >= timestamps[-1]:
            print('Warning:', timestamps[0], timestamps[-1], time)
            return self.xy[-1], self.yaw[-1]

        idx = np.searchsorted(timestamps, time) - 1
        t0, t1 = timestamps[idx], timestamps[idx + 1]
        ratio = (time - t0) / (t1 - t0)

        xy0, xy1 = self.xy[idx], self.xy[idx + 1]
        yaw0, yaw1 = self.yaw[idx], self.yaw[idx + 1]

        # Linear interpolation for x, y
        xy_interp = (1 - ratio) * xy0 + ratio * xy1

        # Ensure yaw interpolation accounts for wrapping
        delta_yaw = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))
        yaw_interp = yaw0 + ratio * delta_yaw

        return xy_interp, yaw_interp
    
    @staticmethod
    def pose_to_matrix(xy, yaw):
        """Convert 2D pose (x, y, yaw) to a 4x4 homogeneous matrix."""
        c, s = np.cos(yaw), np.sin(yaw)
        matrix = np.eye(4)
        matrix[0, 0] = c
        matrix[0, 1] = -s
        matrix[1, 0] = s
        matrix[1, 1] = c
        matrix[0, 3] = xy[0]
        matrix[1, 3] = xy[1]
        return matrix
    
    def relative_transform(self, t0, t1):
        xy0, yaw0 = self.interpolate(t0)
        xy1, yaw1 = self.interpolate(t1)

        T0 = self.pose_to_matrix(xy0, yaw0)
        T1 = self.pose_to_matrix(xy1, yaw1)
        T_rel = np.linalg.inv(T1) @ T0
        
        return T_rel
    
class AccInterpolation:
    #linear in m/s²
    #angular vel in rad/s²

    def __init__(self, timestamps, x, *args):
        self.timestamps = np.array(timestamps)
        if len(args) == 0:
            self.linear = np.array([(d['linear']['x'], d['linear']['y'], d['linear']['z']) for d in x])
            self.angular = np.array([(d['angular']['x'], d['angular']['y'], d['angular']['z']) for d in x])
        else:
            assert len(args) == 1
            self.linear = x
            self.angular = args[0]

    def interpolate(self, time):
        # Ensure time is within bounds
        if time < self.timestamps[0]:
            return self.linear[0], self.angular[0]
        if time > self.timestamps[-1]:
            return self.linear[-1], self.angular[-1]
        
        t0 = self.timestamps[0]
        timestamps = np.array([(t-t0).total_seconds() for t in self.timestamps])
        time = ((time-t0)).total_seconds()

        lin_interp = np.array([
            np.interp(time, timestamps, self.linear[:, i]) for i in range(3)
        ])
        ang_interp = np.array([
            np.interp(time, timestamps, self.angular[:, i]) for i in range(3)
        ])
        return lin_interp, ang_interp

class TwistInterpolation:
    #linear in m/s
    #angular vel in rad/s

    def __init__(self, timestamps, x, *args):
        self.timestamps = np.array(timestamps)
        if len(args) == 0:
            self.linear = np.array([(d['linear']['x'], d['linear']['y'], d['linear']['z']) for d in x])
            self.angular = np.array([(d['angular']['x'], d['angular']['y'], d['angular']['z']) for d in x])
        else:
            assert len(args) == 1
            self.linear = x
            self.angular = args[0]

    def interpolate(self, time):
        # Ensure time is within bounds
        if time < self.timestamps[0]:
            return self.linear[0], self.angular[0]
        if time > self.timestamps[-1]:
            return self.linear[-1], self.angular[-1]

        t0 = self.timestamps[0]
        timestamps = np.array([(t-t0).total_seconds() for t in self.timestamps])
        time = ((time-t0)).total_seconds()

        lin_interp = np.array([
            np.interp(time, timestamps, self.linear[:, i]) for i in range(3)
        ])
        ang_interp = np.array([
            np.interp(time, timestamps, self.angular[:, i]) for i in range(3)
        ])
        return lin_interp, ang_interp

class PreprocessedGPS(PoseInterpolation2D):
    #see gps_imu_kalmanfilter

    def __init__(self, filename):
        with np.load(filename) as d:
            timestamps = d['timestamps']
            states = d['states']
        xy = states[:,:2]
        vxy = states[:,2:4]
        yaw = states[:,4]
        super().__init__(timestamps, xy, yaw)
    
    def __call__(self, *args, **kwds):
        return self

class DataLoader:
    def __init__(self, dirnames, data_generators):
        self.dirnames = dirnames
        self.data_generators = data_generators

    def __iter__(self):
        num = len(self.dirnames)
        for i in range(num):
            dirname_current = self.dirnames[i]
            dirname_next = self.dirnames[min(i+2, num-1)]
            time = string_to_timestamp(dirname_current, isdir=True)
            data = (dg(dirname_next if dg.pre_load else dirname_current) for dg in self.data_generators)
            yield data, time
    
    def __len__(self):
        return len(self.dirnames)
    
def get_dirs(root_path):
    dirnames = np.array([os.path.join(root_path, name) for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))])
    times = np.array([string_to_timestamp(dirname, isdir=True) for dirname in dirnames])
    order = np.argsort(times)
    dirnames = dirnames[order]
    times = times[order]
    return dirnames, times

def parse_pose(data):
    heading = data['heading']
    position = data['position']
    r = R.from_quat([heading['qx'],heading['qy'],heading['qz'],heading['qw']])
    t = [position['x'],position['y'],position['z']]
    return (*t,*r.as_euler('xyz'))

class GPSIMUData:
    def __init__(self):
        pass

    def set_data(self, gps_timestamps, gps_xy, gps_yaw, acc_lin, acc_ang, twist_lin, twist_ang):
        self.timestamps = gps_timestamps
        self.gps_xy = gps_xy
        self.gps_yaw = gps_yaw
        self.acc_lin = acc_lin
        self.acc_ang = acc_ang
        self.twist_lin = twist_lin
        self.twist_ang = twist_ang

    def axis_align_gps(self, yaw=None):
        t = self.gps_xy[:1]
        yaw = self.gps_yaw[0] if yaw is None else yaw
        c = np.cos(yaw)
        s = np.sin(yaw)
        R = np.array([[c,-s],[s,c]])
        self.gps_xy = (self.gps_xy - t) @ R
        self.gps_yaw = ((self.gps_yaw - yaw + np.pi) % (2*np.pi)) - np.pi

    def load(self, dirnames):
        self.set_data(*collect_all_data(dirnames))

    def __iter__(self):
        for i in range(len(self)):
            yield self.timestamps[i], (self.gps_xy[i], self.gps_yaw[i,None], self.twist_lin[i], self.twist_ang[i]), (self.acc_lin[i], self.acc_ang[i])

    def __len__(self):
        return len(self.timestamps)
    
    def __getitem__(self, idx):
        return self.timestamps[idx], (self.gps_xy[idx], self.gps_yaw[idx,None], self.twist_lin[idx], self.twist_ang[idx]), (self.acc_lin[idx], self.acc_ang[idx])
    
    def interpolate(self, time):
        gps_interp = PoseInterpolation2D(self.timestamps, self.gps_xy, self.gps_yaw)
        twist_interp = TwistInterpolation(self.timestamps, self.twist_lin, self.twist_ang)
        acc_interp = AccInterpolation(self.timestamps, self.acc_lin, self.acc_ang)
        gps_xy, gps_yaw = gps_interp.interpolate(time)
        twist_lin, twist_ang = twist_interp.interpolate(time)
        acc_lin, acc_ang = acc_interp.interpolate(time)
        return (gps_xy, gps_yaw[None], twist_lin, twist_ang), (acc_lin, acc_ang)



def collect_all_data(dirnames):
    all_data = {}

    gps_loader = IMULoader()

    for dirname in dirnames:
        gps, acc, twist = gps_loader(dirname)
        for t, xy, yaw, acc_lin, acc_ang, twist_lin, twist_ang in zip(gps.timestamps, gps.xy, gps.yaw, acc.linear, acc.angular, twist.linear, twist.angular):
            all_data[t] = (xy, yaw, acc_lin, acc_ang, twist_lin, twist_ang)  # overwrite if t repeats

    # Sort by timestamp
    sorted_items = sorted(all_data.items())
    sorted_timestamps = np.array([t for t, _ in sorted_items])
    sorted_xy = np.array([val[0] for _, val in sorted_items])
    sorted_yaw = np.array([val[1] for _, val in sorted_items])
    sorted_acc_lin = np.array([val[2] for _, val in sorted_items])
    sorted_acc_ang = np.array([val[3] for _, val in sorted_items])
    sorted_twist_lin = np.array([val[4] for _, val in sorted_items])
    sorted_twist_ang = np.array([val[5] for _, val in sorted_items])

    return (sorted_timestamps, sorted_xy, sorted_yaw, sorted_acc_lin, sorted_acc_ang, sorted_twist_lin, sorted_twist_ang)


def point_cloud_features(pc_load: o3d.geometry.PointCloud, color_channel: int=None, color_scale=255.):
    points = np.asarray(pc_load.points, dtype=np.float64)
    if pc_load.has_colors() and color_channel is not None:
        if color_channel == 3:
            colors = np.asarray(pc_load.colors) * color_scale
            points = np.hstack((points, colors))
        elif color_channel == 1:
            colors = np.asarray(pc_load.colors) * color_scale
            points = np.hstack((points, colors[:,:1])) # ignore other channels
        else:
            raise Exception()
    return points