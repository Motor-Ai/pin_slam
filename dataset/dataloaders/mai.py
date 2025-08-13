import numpy as np
from pathlib import Path
from scipy.linalg import logm, expm
from ..utils.data_io import get_dirs, GPSIMUData, PointCloudsLoader, PoseInterpolation2D, point_cloud_features
from ..utils.pose_kalmanfilter import PoseKalmanFilter
from ..utils.ros_io import PointCloudLoaderRos

def parse_config(lidar_calib_file, pc_filenames = ['ivu2'], color_channel=None, guessing='default', ros_load=False, interval=1):
    assert len(pc_filenames) == 1, 'not implemented'
    return lidar_calib_file, pc_filenames, color_channel, guessing, ros_load, interval

class MAIDataset:
    def __init__(self, data_dir: Path, sequence, *args, **kwargs):
        lidar_calib_file, pc_filenames, self.color_channel, guessing, ros_load, interval = parse_config(**sequence)
        if interval != 1:
            assert ros_load, 'only for ros_load available'

        if ros_load:
            assert len(pc_filenames) == 1, 'only one sensor supported so far'
            self.pc_loader = PointCloudLoaderRos(data_dir, topic_name='/' + pc_filenames[0], prescan=False, extrinsics_file=lidar_calib_file, pc_o3d=True, verbose=False, interval=interval)
        else:
            # find folders
            root_frames = data_dir
            dirnames, _ = get_dirs(root_frames)
            self.pc_loader = PointCloudsLoader(pc_filenames, lidar_calib_file, dirnames=dirnames)
        print('number of frames:', len(self.pc_loader))

        # complete transformation to base frame needs to be done after SLAM.
        # PIN expects data to move primarily in x-y-plane and z-axis for height estimations.
        # therefore we only apply rotations and no translations -> rays are still calculated correctly
        # after slam is done translation is added.
        assert len(pc_filenames) == 1, 'only one sensor supported so far'
        calib = self.pc_loader.extrinsics[pc_filenames[0]]
        calib_rotation_only = np.eye(4)
        calib_rotation_only[:3,:3] = calib[:3,:3]
        calib_translation_only = np.eye(4)
        calib_translation_only[:3,3] = calib[:3,3]
        self.calibration = dict(Tr=calib_translation_only[:3,:4]) # is used in slam_dataset.py at the end
        calib[:] = calib_rotation_only # we only apply rotation - it does not effect rays
        
        self.actual_timestamps = dict()
        if guessing == 'default':
            self.initial_guess = self.initial_guess_default
        elif guessing == 'gps':
            assert isinstance(self.pc_loader, PointCloudsLoader), 'not implemented for ROS inputs yet'
            self.initial_guess = self.initial_guess_gps
            self.gps_imu_data = GPSIMUData()
            self.gps_imu_data.load(dirnames)
            print('gps & imu loaded')
            self.gps_interp = PoseInterpolation2D(self.gps_imu_data.timestamps, self.gps_imu_data.gps_xy, self.gps_imu_data.gps_yaw)
        elif guessing == 'time':
            self.initial_guess = self.initial_guess_time
        elif guessing == 'kf':
            self.initial_guess = self.initial_guess_kf
            self.kf = PoseKalmanFilter()
        else:
            raise Exception()

    def __getitem__(self, idx):
        timestamp, pcd = self.pc_loader[idx]
        if isinstance(timestamp, list):
            assert len(timestamp) == 1
            timestamp = timestamp[0]
        if isinstance(pcd, list):
            assert len(pcd) == 1
            pcd = pcd[0]

        #import open3d as o3d
        #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,0])
        #o3d.visualization.draw_geometries([pcd, mesh_frame])

        self.actual_timestamps[idx] = timestamp
        points = point_cloud_features(pcd, color_channel=self.color_channel, color_scale=1.0)
        point_ts = None
        frame_data = {"points": points, "point_ts": point_ts}
        return frame_data

    def __len__(self):
        return len(self.pc_loader)
    
    def initial_guess_default(self, frame_id: int, last_pose_ref: np.ndarray, last_odom_tran: np.ndarray):
        #print_trans('guess', last_odom_tran)
        return last_pose_ref @ last_odom_tran # T_world<-cur = T_world<-last @ T_last<-cur
    
    def initial_guess_time(self, frame_id: int, last_pose_ref: np.ndarray, last_odom_tran: np.ndarray):
        t0 = self.actual_timestamps.get(frame_id-2, None)
        t1 = self.actual_timestamps[frame_id-1]
        t2 = self.actual_timestamps[frame_id]
        if t0 is None:
            last_odom_tran_new = last_odom_tran
        else:
            #SE(3) extrapolation using time
            last_odom_tran_new = extrapolate_se3(last_odom_tran, t1-t0, t2-t1)
        #print_trans('guess', last_odom_tran)
        #print_trans('time guess', last_odom_tran_new)
        #print(f'delta time: {t2-t1}' if t0 is None else f'delta times: {t1-t0}, {t2-t1}')
        return last_pose_ref @ last_odom_tran_new # T_world<-cur = T_world<-last @ T_last<-cur

    def initial_guess_gps(self, frame_id: int, last_pose_ref: np.ndarray, last_odom_tran: np.ndarray):
        time_last = self.actual_timestamps[frame_id-1]
        time_cur = self.actual_timestamps[frame_id]
        #relative transformation gps_last <- gps_cur
        last_odom_tran_new = self.gps_interp.relative_transform(time_cur, time_last)
        #print_trans('guess', last_odom_tran)
        #print_trans('gps guess', last_odom_tran_new)
        return last_pose_ref @ last_odom_tran_new # T_world<-cur = T_world<-last @ T_last<-cur
    
    def initial_guess_kf(self, frame_id: int, last_pose_ref: np.ndarray, last_odom_tran: np.ndarray):
        t0 = self.actual_timestamps.get(frame_id-2, None)
        t1 = self.actual_timestamps[frame_id-1]
        t2 = self.actual_timestamps[frame_id]

        if t0 is None:
            last_odom_tran_new = last_odom_tran
        else:
            self.kf.predict(t1-t0)
            self.kf.update(last_odom_tran)
            last_odom_tran_new = self.kf.predict_future_pose(t2-t1)
        #print_trans('guess', last_odom_tran)
        #print_trans('kf guess', last_odom_tran_new)
        return last_pose_ref @ last_odom_tran_new # T_world<-cur = T_world<-last @ T_last<-cur
    

def extrapolate_se3(m: np.ndarray, dt1: float, dt2: float) -> np.ndarray:
    # Logarithm of transformation (Lie algebra element)
    xi = logm(m)
    # Scale the twist by the ratio of time deltas
    scale = dt2 / dt1
    xi_scaled = scale * xi
    # Exponentiate to get the new transformation
    m_extrapolated = expm(xi_scaled)
    return m_extrapolated

def print_trans(name, odom_tran):
    from scipy.spatial.transform import Rotation as R
    xyz = tuple(odom_tran[:3, 3])
    rpy = tuple(R.from_matrix(odom_tran[:3,:3]).as_euler('xyz', degrees=True))
    print(
        f'{name}: translation=[{xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f}] m, '
        f'rotation=[{rpy[0]:.3f} {rpy[1]:.3f} {rpy[2]:.3f}] deg'
    )