import numpy as np
import open3d as o3d
from numba import njit, types, typed

class VoxelMerger:
    def __init__(self, voxel_size=0.05, origin=np.zeros(3)):
        self.voxel_size = float(voxel_size)
        self.origin = np.asarray(origin, dtype=np.float64)

        key_type = types.UniTuple(types.int64, 3)
        val_type = types.float64[:]
        self._voxels = typed.Dict.empty(key_type, val_type)

    @staticmethod
    @njit
    def _add_points_direct(pts, cols, origin, voxel_size, gdict):
        n = pts.shape[0]
        for i in range(n):
            # Compute voxel key
            key = (
                int(np.floor((pts[i, 0] - origin[0]) / voxel_size)),
                int(np.floor((pts[i, 1] - origin[1]) / voxel_size)),
                int(np.floor((pts[i, 2] - origin[2]) / voxel_size)),
            )

            if key in gdict:
                val = gdict[key]
                val[0] += pts[i, 0]
                val[1] += pts[i, 1]
                val[2] += pts[i, 2]
                val[3] += cols[i, 0]
                val[4] += cols[i, 1]
                val[5] += cols[i, 2]
                val[6] += 1.0
                gdict[key] = val
            else:
                gdict[key] = np.array(
                    [pts[i, 0], pts[i, 1], pts[i, 2], cols[i, 0], cols[i, 1], cols[i, 2], 1.0],
                    dtype=np.float64,
                )

    @staticmethod
    @njit
    def _finalize(gdict):
        n = len(gdict)
        pts_out = np.empty((n, 3), dtype=np.float64)
        cols_out = np.empty((n, 3), dtype=np.float64)

        i = 0
        for key in gdict:
            v = gdict[key]
            cnt = v[6]
            pts_out[i, 0] = v[0] / cnt
            pts_out[i, 1] = v[1] / cnt
            pts_out[i, 2] = v[2] / cnt
            cols_out[i, 0] = v[3] / cnt
            cols_out[i, 1] = v[4] / cnt
            cols_out[i, 2] = v[5] / cnt
            i += 1

        return pts_out, cols_out

    def add_point_cloud(self, pc: o3d.geometry.PointCloud):
        pts = np.asarray(pc.points, dtype=np.float64)
        if pc.has_colors():
            cols = np.asarray(pc.colors, dtype=np.float64)
        else:
            cols = np.zeros_like(pts, dtype=np.float64)

        self._add_points_direct(pts, cols, self.origin, self.voxel_size, self._voxels)

    def final_point_cloud(self):
        pts_out, cols_out = self._finalize(self._voxels)
        final_pc = o3d.geometry.PointCloud()
        final_pc.points = o3d.utility.Vector3dVector(pts_out)
        final_pc.colors = o3d.utility.Vector3dVector(cols_out)
        return final_pc

