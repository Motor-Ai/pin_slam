import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import inv, expm

def hat_se3(omega, v):
    """Construct the 4x4 matrix for the twist [v, ω]."""
    wx, wy, wz = omega
    vx, vy, vz = v
    omega_hat = np.array([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ])
    mat = np.zeros((4, 4))
    mat[:3, :3] = omega_hat
    mat[:3, 3] = v
    return mat


class PoseKalmanFilter:
    def __init__(self,
                 p_uncertainty=1e-3,
                 q_t_uncertainty=1e-3,
                 q_r_uncertainty=1e-3,
                 v_uncertainty_fact=10.,
                 yaw_uncertainty_fact=10.):
        # State: [x, y, z, roll, pitch, yaw, vx, vy, vz, v_roll, v_pitch, v_yaw]
        self.x = np.zeros((12, 1))
        self.P = np.eye(12) * p_uncertainty  # Initial state covariance
        # Process noise for positions and orientations
        q_translation = [q_t_uncertainty] * 3
        q_rotation = [q_r_uncertainty] * 2 + [q_r_uncertainty * yaw_uncertainty_fact]
        # Process noise for velocities (higher uncertainty)
        q_translation_vel = [q_t_uncertainty * v_uncertainty_fact] * 3
        q_rotation_vel = [q_r_uncertainty * v_uncertainty_fact] * 2 + [q_r_uncertainty * v_uncertainty_fact * yaw_uncertainty_fact]
        # Full process noise
        self.Q = np.diag(q_translation + q_rotation + q_translation_vel + q_rotation_vel)

    def predict(self, delta_time):
        dt = delta_time.total_seconds()
        F = np.eye(12)
        F[0, 6] = dt
        F[1, 7] = dt
        F[2, 8] = dt
        F[3, 9] = dt
        F[4, 10] = dt
        F[5, 11] = dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def predict_future_pose(self, delta_time, method='rigid'):
        if method == 'rigid':
            return self.predict_future_pose_rigid(delta_time)
        if method == 'simple':
            return self.predict_future_pose_simple(delta_time)
        raise Exception(f'Unknown method: {method}')

    def predict_future_pose_rigid(self, delta_time):
        dt = delta_time.total_seconds()
        v = self.x[6:9, 0]    # body-frame linear velocity
        omega = self.x[9:12, 0]  # body-frame angular velocity

        twist_hat = hat_se3(omega, v)
        T_rel = expm(twist_hat * dt)

        #return T_rel

        # Current pose
        pos = self.x[:3, 0]
        rot_euler = self.x[3:6, 0]
        T_current = np.eye(4)
        T_current[:3, :3] = R.from_euler('xyz', rot_euler).as_matrix()
        T_current[:3, 3] = pos
        T_new = T_current @ T_rel
        return T_new
    
    def predict_future_pose_simple(self, delta_time):
        dt = delta_time.total_seconds()
        pos = self.x[:3, 0]
        rot = self.x[3:6, 0]
        vel = self.x[6:9, 0]
        ang_vel = self.x[9:12, 0]
        pos_new = pos + vel * dt
        rot_new = rot + ang_vel * dt
        return state_to_matrix(pos_new, rot_new)

    def update(self, T_icp, info_icp=None, measurement_noise=1e-2):
        # Extract translation and rotation (roll, pitch, yaw)
        t, angles = matrix_to_state(T_icp)

        # Measurement vector: position and rotation
        z = np.hstack([t, angles]).reshape(6, 1)

        # Measurement matrix: we observe only positions and rotations, not velocities
        H = np.zeros((6, 12))
        H[:6, :6] = np.eye(6)

        # Measurement covariance
        if info_icp is not None:
            cov_icp = np.linalg.inv(info_icp)
        else:
            # Use default measurement noise (e.g., identity scaled)
            cov_icp = np.eye(6) * measurement_noise

        # Kalman Gain
        S = H @ self.P @ H.T + cov_icp
        K = self.P @ H.T @ np.linalg.inv(S)

        # Innovation
        y = z - H @ self.x

        # Update state and covariance
        self.x = self.x + K @ y
        self.P = (np.eye(12) - K @ H) @ self.P


    def get_pose(self, as_matrix=False):
        x = self.x[:6].flatten()  # [x, y, z, roll, pitch, yaw]
        if as_matrix:
            return state_to_matrix(x[:3], x[3:])
        return x

    def get_velocity(self):
        return self.x[6:].flatten()  # velocities
    
    def get_pose_uncertainty(self):
        # 6x6 covariance for [x, y, z, roll, pitch, yaw]
        return self.P[:6, :6]

    def get_velocity_uncertainty(self):
        # 6x6 covariance for [vx, vy, vz, v_roll, v_pitch, v_yaw]
        return self.P[6:, 6:]


def matrix_to_state(matrix):
    t = matrix[:3, 3]
    Rot = matrix[:3, :3]
    angles = R.from_matrix(Rot).as_euler('xyz', degrees=False)
    return t, angles

def state_to_matrix(translation, angles):
    T = np.eye(4)
    T[:3, 3] = translation
    T[:3, :3] = R.from_euler('xyz', angles).as_matrix()
    return T