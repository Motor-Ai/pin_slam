import os, sys
import numpy as np
from datetime import datetime
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import time
sys.path.insert(0,'.')
from dataset.utils.ros_io import PointCloudLoaderRos
from dataset.utils.voxelization import VoxelMerger


pc_path = "/home/thomas/Downloads/session_01/bag_2025_07_23_07_34_37/"
lidar_calib_file = "/home/thomas/Downloads/exp_data_collection_test/calibration/lidars.yaml"
pc_loader = PointCloudLoaderRos(pc_path, '/ivu2', extrinsics_file=lidar_calib_file)
print('pc loader ready')
#transforms_path = '/home/thomas/Documents/coding/lidar_cam_metric/PIN_SLAM/runs/schwarzerberg/loader_ivu_color_2025-08-11_09-46-52/slam_poses_kitti.txt'
#transforms_path = '/home/thomas/Documents/coding/lidar_cam_metric/PIN_SLAM/runs/schwarzerberg/loader_ivu_color_2025-08-11_09-46-52/odom_poses_kitti.txt'
#transforms_path = '/home/thomas/Documents/coding/lidar_cam_metric/PIN_SLAM/runs/schwarzerberg/loader_ivu_color_2025-08-11_14-19-08/slam_poses_kitti.txt'
transforms_path = '/home/thomas/Documents/coding/lidar_cam_metric/PIN_SLAM/runs/schwarzerberg/loader_ivu_color2_2025-08-12_11-50-53/slam_poses_kitti.txt'


def make_arrows(transforms):
    # Create arrows transformed by each matrix
    arrows = []
    R_z_to_x = o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi / 2, 0])
    for T in transforms:
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.02,
            cone_radius=0.04,
            cylinder_height=0.3,
            cone_height=0.1
        )
        arrow.rotate(R_z_to_x, center=(0, 0, 0))
        arrow.paint_uniform_color([1, 0, 0])  # red arrow
        arrow.transform(T)
        arrows.append(arrow)
    return arrows

def load_kitti_poses(file_path):
    """Load a list of 3x4 transformation matrices from a KITTI-format text file."""
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            T = np.fromstring(line, sep=' ').reshape((3, 4))
            T_hom = np.vstack((T, [0, 0, 0, 1]))  # Convert to 4x4
            poses.append(T_hom)
    return poses

def filter_point_cloud(pcd, x_range=(0, 50), y_range=(-20, 20)):
    points = np.asarray(pcd.points)
    
    # Create mask for x and y ranges
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    )

    # Apply mask
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])

    # Optional: retain colors or other attributes if present
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        filtered_pcd.normals = o3d.utility.Vector3dVector(normals[mask])

    return filtered_pcd


def merge_voxelized(pc_loader, poses, num_point_clouds=None, voxel_size = 0.1, origin = np.array([0.0, 0.0, 0.0])):
    if num_point_clouds is None:
        num_point_clouds = len(pc_loader)
    vm = VoxelMerger(voxel_size=voxel_size, origin=origin)
    # loop
    for i in tqdm(range(num_point_clouds)):  # Replace with your actual number of clouds
        timestamp, pc = pc_loader[i]
        pc = pc.transform(poses[i])
        vm.add_point_cloud(pc)
        if i % 100 == 0:
            print('number of voxels:', len(vm._voxels))

    # Convert accumulated voxels to final point cloud
    final_pc = vm.final_point_cloud()
    return final_pc






# list of 4x4 transformation matrices
if transforms_path is not None:
    transforms = load_kitti_poses(transforms_path)
    arrows = make_arrows(transforms)
    clear_geoms = False
else:
    transforms = None
    arrows = None
    clear_geoms = True


if True:
    # voxelize data and save and visualize it

    # write
    #pcd = merge_voxelized(pc_loader, transforms, num_point_clouds=len(pc_loader))
    #o3d.io.write_point_cloud(os.path.join(os.path.dirname(transforms_path), "merged_downsampled_slam.ply"), pcd)

    # re-load
    pcd = o3d.io.read_point_cloud(os.path.join(os.path.dirname(transforms_path), "merged_downsampled_slam.ply"))

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,0])
    o3d.visualization.draw_geometries([pcd, mesh_frame]+arrows)
    quit()


## visualize trip - animated!

offset = 0
num = 9999
tw = 0.0


num = min(num, len(pc_loader))
pc_iter = iter(pc_loader)
idx = [0]
progressbar = tqdm(total=num)

def get_next_pc():
    global pc_iter
    i = idx[0]
    timestamp, pcd = next(pc_iter)
    pcd = filter_point_cloud(pcd)
    if transforms is not None:
        pcd = pcd.transform(transforms[i])
    return pcd

def animation_callback(vis):
    i = idx[0]
    if i >= num:
        return False  # stop the animation
    progressbar.update(1)
    if clear_geoms:
        vis.clear_geometries()
    pcd = get_next_pc()
    vis.add_geometry(pcd)
    if arrows is not None:
        vis.add_geometry(arrows[i+offset])
    idx[0] += 1
    time.sleep(tw)  # Control frame rate
    return True  # continue the animation

o3d.visualization.draw_geometries_with_animation_callback([get_next_pc()], animation_callback)