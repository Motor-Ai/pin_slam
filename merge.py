import os
import numpy as np
import open3d as o3d

def load_kitti_poses(file_path):
    """Load a list of 3x4 transformation matrices from a KITTI-format text file."""
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            T = np.fromstring(line, sep=' ').reshape((3, 4))
            T_hom = np.vstack((T, [0, 0, 0, 1]))  # Convert to 4x4
            poses.append(T_hom)
    return poses

def transform_ply(input_path, transformation):
    """Load and transform a .ply file using a 4x4 transformation matrix."""
    pcd = o3d.io.read_point_cloud(input_path)
    pcd.transform(transformation)
    return pcd

def save_ply(pcd, output_path):
    """Save a point cloud to a .ply file."""
    o3d.io.write_point_cloud(output_path, pcd)

def merge_folders(input_folders, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Output containers
    merged_poses = []
    merged_odom_pcd = o3d.geometry.PointCloud()
    merged_mesh_pcd = o3d.geometry.PointCloud()
    merged_map_pcd = o3d.geometry.PointCloud()

    global_transform = np.eye(4)

    for i, folder in enumerate(input_folders):
        pose_file = os.path.join(folder, 'odom_poses_kitti.txt')
        poses = load_kitti_poses(pose_file)

        #print(np.array(poses)[:,:2,3])

        # Compute absolute poses using global_transform
        abs_poses = [global_transform @ pose for pose in poses]

        # Append poses in global frame (omit last frame if not final folder)
        if i < len(input_folders) - 1:
            abs_poses = abs_poses[:-1]
        merged_poses.extend(abs_poses)

        # Transform and merge point clouds
        odom_pcd = transform_ply(os.path.join(folder, 'odom_poses.ply'), global_transform)
        mesh_pcd = transform_ply(os.path.join(folder, 'mesh/mesh_24cm.ply'), global_transform)
        map_pcd = transform_ply(os.path.join(folder, 'map/neural_points.ply'), global_transform)

        merged_odom_pcd += odom_pcd
        merged_mesh_pcd += mesh_pcd
        merged_map_pcd += map_pcd

        # Update global transform to last pose of current folder
        global_transform = abs_poses[-1]

    # Save merged outputs
    np.savetxt(os.path.join(output_dir, 'merged_poses_kitti.txt'),
               np.array([pose[:3].reshape(-1) for pose in merged_poses]), fmt='%.6f')

    save_ply(merged_odom_pcd, os.path.join(output_dir, 'merged_odom_poses.ply'))
    save_ply(merged_mesh_pcd, os.path.join(output_dir, 'merged_mesh_24cm.ply'))
    save_ply(merged_map_pcd, os.path.join(output_dir, 'merged_neural_points.ply'))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge odometry folders into a global map.")
    parser.add_argument("folders", nargs='+', help="Input folders to merge (in order).")
    parser.add_argument("--output", required=True, help="Directory to write merged outputs.")
    args = parser.parse_args()

    merge_folders(args.folders, args.output)
