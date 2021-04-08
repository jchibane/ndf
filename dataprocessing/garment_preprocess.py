from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
from glob import glob
import os
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import random
import traceback
from functools import partial
import pymesh


def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


def voxelized_pointcloud_boundary_sampling(path, sigmas, res, inp_points, sample_points):
    try:
        out_voxelization_file = path + '/voxelized_point_cloud_{}res_{}points.npz'.format(res, inp_points)

        off_path = path + '/mesh.off'
        mesh = trimesh.load(off_path)
        py_mesh = pymesh.load_mesh(off_path)

        # =====================
        # Voxelized point cloud
        # =====================

        if not os.path.exists(out_voxelization_file):

            bb_min = -0.5
            bb_max = 0.5

            point_cloud = mesh.sample(inp_points)

            # Grid Points used for computing occupancies
            grid_points = create_grid_points_from_bounds(bb_min, bb_max, args.res)

            # KDTree creation for fast querying nearest neighbour to points on the point cloud
            kdtree = KDTree(grid_points)
            _, idx = kdtree.query(point_cloud)

            occupancies = np.zeros(len(grid_points), dtype=np.int8)
            occupancies[idx] = 1
            compressed_occupancies = np.packbits(occupancies)

            np.savez(out_voxelization_file, point_cloud=point_cloud, compressed_occupancies = compressed_occupancies,
                     bb_min = bb_min, bb_max = bb_max, res = res)
            print('Finished Voxelized point cloud {}'.format(path))

        # ==================
        # Boundary Sampling
        # ==================
        for sigma in sigmas:
            out_sampling_file = path + '/pymesh_boundary_{}_samples.npz'.format( sigma)

            if not os.path.exists(out_sampling_file):
                points = mesh.sample(sample_points)
                if sigma == 0:
                    boundary_points = points
                else:
                    boundary_points = points + sigma * np.random.randn(sample_points, 3)

                # Transform the boundary points to grid coordinates
                grid_coords = boundary_points.copy()
                grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

                grid_coords = 2 * grid_coords

                # distance field calculation
                if sigma == 0:
                    df = np.zeros(boundary_points.shape[0])
                else:
                    df = np.sqrt(pymesh.distance_to_mesh(py_mesh, boundary_points)[0])
                np.savez(out_sampling_file, points=boundary_points, df=df, grid_coords=grid_coords)

                print('Finished boundary sampling {}'.format(out_sampling_file))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )

    parser.add_argument("--input_path", type = str)
    parser.add_argument("--sigmas", nargs = '+', type = float)
    parser.add_argument("--res", type = int)
    parser.add_argument("--inp_points", type = int)
    parser.add_argument("--sample_points", type = int)

    args = parser.parse_args()


    paths = glob(args.input_path + '/*/')

    #To run te script multiple times in parallel: shuffling the data
    random.shuffle(paths)
    print(mp.cpu_count())
    p = Pool(mp.cpu_count())
    p.map(partial(voxelized_pointcloud_boundary_sampling, sigmas=args.sigmas, res = args.res,
                  inp_points = args.inp_points, sample_points = args.sample_points), paths)
    p.close()
    p.join()
