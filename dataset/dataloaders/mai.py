import numpy as np
from scipy.linalg import logm, expm
from ..utils.data_io import get_dirs, GPSIMUData, PointCloudsLoader, PoseInterpolation2D, point_cloud_features
from ..utils.pose_kalmanfilter import PoseKalmanFilter

def parse_config(lidar_calib_file, pc_filenames = ['ivu2'], color_channel=None, guessing='default'):
    assert len(pc_filenames) == 1, 'not implemented'
    return lidar_calib_file, pc_filenames, color_channel, guessing

class MAIDataset:
    def __init__(self, data_dir, sequence, *args, **kwargs):
        root_frames = data_dir
        lidar_calib_file, pc_filenames, self.color_channel, guessing = parse_config(**sequence)

        # find folders
        dirnames, _ = get_dirs(root_frames)
        print('number of frames:', len(dirnames))
        self.pc_loader = PointCloudsLoader(pc_filenames, lidar_calib_file, dirnames=dirnames)
        print('pc loader ready')

        # transformation to base frame needs to be done after SLAM!
        assert len(self.pc_loader.pc_names) == 1, 'only one sensor supported so far'
        calib = self.pc_loader.extrinsics[self.pc_loader.pc_names[0]]
        self.calibration = dict(Tr=calib[:3,:4]) # is used in slam_dataset.py at the end
        self.pc_loader.extrinsics = None  # transformations disabled
        
        self.actual_timestamps = dict()
        if guessing == 'default':
            self.initial_guess = self.initial_guess_default
        elif guessing == 'gps':
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
        timestamps, pcds = self.pc_loader[idx]
        assert len(pcds) == 1
        pcd = pcds[0]
        timestamp = timestamps[0]
        self.actual_timestamps[idx] = timestamp
        points = point_cloud_features(pcd, color_channel=self.color_channel, color_scale=1.0)
        point_ts = None
        frame_data = {"points": points, "point_ts": point_ts}
        return frame_data

    def __len__(self):
        return len(self.pc_loader)
    
    def initial_guess_default(self, frame_id: int, last_pose_ref: np.ndarray, last_odom_tran: np.ndarray):
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
        #print('guess: [%.3f %.3f %.3f]' % tuple(last_odom_tran[:3, 3]))
        #print('time guess: [%.3f %.3f %.3f]' % tuple(last_odom_tran_new[:3, 3]))
        return last_pose_ref @ last_odom_tran_new # T_world<-cur = T_world<-last @ T_last<-cur

    def initial_guess_gps(self, frame_id: int, last_pose_ref: np.ndarray, last_odom_tran: np.ndarray):
        time_last = self.actual_timestamps[frame_id-1]
        time_cur = self.actual_timestamps[frame_id]
        #relative transformation gps_last <- gps_cur
        last_odom_tran_new = self.gps_interp.relative_transform(time_cur, time_last)
        #print('guess: [%.3f %.3f %.3f]' % tuple(last_odom_tran[:3, 3]))
        #print('gps guess: [%.3f %.3f %.3f]' % tuple(last_odom_tran_new[:3, 3]))
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
        #print('guess: [%.3f %.3f %.3f]' % tuple(last_odom_tran[:3, 3]))
        #print('time guess: [%.3f %.3f %.3f]' % tuple(last_odom_tran_new[:3, 3]))
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
