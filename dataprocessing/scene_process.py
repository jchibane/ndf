import trimesh
import pymesh
import numpy as np

import os
import traceback
from functools import partial
from scipy.spatial import cKDTree as KDTree
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from glob import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import random


def find_verts(verts, minx, maxx, miny, maxy, minz, maxz):
    verts_locs = np.where(verts[:, 0] <= maxx)
    verts_loc2 = np.where(verts[:, 0] >= minx)

    verts_locs3 = np.where(verts[:,1] <= maxy)
    verts_locs4 = np.where(verts[:,1] >= miny)

    verts_locs5 = np.where(verts[:,2] <= maxz)
    verts_locs6 = np.where(verts[:,2] >= minz)

    ret_verts = list(set(verts_locs[0].tolist()) & (set(verts_loc2[0].tolist())) & (set(verts_locs3[0].tolist())) & set(verts_locs4[0].tolist()) & set(verts_locs5[0].tolist()) & (set(verts_locs6[0].tolist())))

    return ret_verts

def get_boxes(points, low, high):
    GRID_sIZE = 2.5

    delt = -0.1
    dict_ret = {}
    xs = np.linspace(-53 + delt, 72 + delt, np.uint8(np.round((72+53)/ GRID_sIZE )) + 1)
    ys = np.linspace(-51 + delt, 54 + delt, np.uint8(np.round((105 / GRID_sIZE))) + 1)
    zs = np.linspace(-7.5 +delt, 12.5 + delt, np.uint8(np.round((20/ GRID_sIZE))) + 1)


    index = 0
    for i in range(len(xs) - 1):
        for j in range(len(ys) -1):
            for k in range(len(zs) - 1):
                index = index + 1
                x_min = xs[i]
                x_max =  xs[i+1]

                y_min = ys[j]
                y_max = ys[j+1]

                z_min = zs[k]
                z_max = zs[k + 1]

                if x_min >= high[0] or x_max <= low[0] or y_min >= high[1] or y_max <= low[1] or z_min >= high[2] or z_max <= low[2]:
                    continue
                else:
                    verts_inds = find_verts(points, x_min, x_max, y_min, y_max, z_min, z_max)
                    if not len(verts_inds) == 0:
                        dict_ret[(x_min, y_min, z_min)] = verts_inds

    return dict_ret


def create_grid_points_from_bounds(min_x, max_x, min_y, max_y, min_z, max_z, res):
    x = np.linspace(min_x, max_x, res)
    y = np.linspace(min_y, max_y, res)
    z = np.linspace(min_z, max_z, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij', sparse=False)
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list

def bd_sampl_vx_ptcld(input_path, output_path, sigmas, res, density):

    print('Start with: ', input_path)
    try:
        norm_path = os.path.normpath(input_path)
        scan_name = norm_path.split(os.sep)[-1]

        obj_path = input_path + '/{}_mesh_texture.obj'.format(scan_name)

        # Check if some other process already working on this path
        out_query = output_path + '/{}/*/pymesh_boundary_{}_samples.npz'.format(scan_name, sigmas[0])
        if len(glob(out_query)) > 0:
            print('Exists - skip!')
            return

        mesh = trimesh.load(obj_path)
        py_mesh = pymesh.load_mesh(obj_path)

        # Input Point Cloud
        sample_num = int(mesh.area / density)
        point_cloud = mesh.sample(sample_num)

        # Boundary sampling points
        print('Random sample mesh')
        sample_num_bd = 5 * sample_num
        points = mesh.sample(sample_num_bd)

        boundary_points_list = []
        df_list = []
        # ==============================
        #   Distance Field Computation
        # ==============================
        for sigma in sigmas:
            print('Distance computation sigma {}'.format(sigma))

            boundary_points = points + sigma * np.random.randn(sample_num_bd, 3)
            boundary_points_list.append(boundary_points)
            df_list.append(np.sqrt(pymesh.distance_to_mesh(py_mesh, boundary_points)[0]))

        print('Split into chunks: ', input_path)
        split_dict = get_boxes(point_cloud, *mesh.bounds)

        for cube_corner in split_dict:
            print('Start voxelization: ', input_path)

            # ===========================
            #    Voxelized Point Cloud
            # ===========================
            out_cube_path = output_path + '/{}/{}/'.format(scan_name, cube_corner)
            os.makedirs(out_cube_path, exist_ok=True)
            out_file = out_cube_path + 'voxelized_point_cloud_{}res_{}density.npz'.format(res, density)

            min_x, min_y, min_z  = cube_corner
            verts_inds = split_dict[cube_corner]
            voxel_point_cloud = point_cloud[verts_inds]

            grid_points = create_grid_points_from_bounds(min_x, min_x + 2.5, min_y, min_y + 2.5, min_z, min_z + 2.5, res)
            occupancies = np.zeros(len(grid_points), dtype=np.int8)
            kdtree = KDTree(grid_points)
            _, idx = kdtree.query(voxel_point_cloud)
            occupancies[idx] = 1

            compressed_occupancies = np.packbits(occupancies)

            np.savez(out_file, point_cloud=voxel_point_cloud, compressed_occupancies=compressed_occupancies, res=res)

            # =====================================
            #   Split Distance Field into Cubes
            # =====================================
            print('Start corner df computation: ', input_path)

            for i, sigma in enumerate(sigmas):

                df = df_list[i]
                boundary_points = boundary_points_list[i]

                verts_inds = find_verts(boundary_points, min_x, min_x + 2.5, min_y, min_y + 2.5, min_z, min_z + 2.5)

                cube_df = df[verts_inds]
                cube_points = boundary_points[verts_inds]

                cube_points2 = cube_points[:] - cube_corner
                grid_cube_points =cube_points2.copy()
                grid_cube_points[:, 0], grid_cube_points[:, 2] = cube_points2[:, 2], cube_points2[:, 0]
                grid_cube_points = grid_cube_points / 2.5
                grid_cube_points = 2 * grid_cube_points - 1

                out_path = output_path + '/{}/{}/'.format(scan_name, cube_corner)
                os.makedirs(out_path, exist_ok=True)

                np.savez(out_path + '/pymesh_boundary_{}_samples.npz'.format(sigma), points=cube_points, df = cube_df, grid_coords= grid_cube_points)

        print('Finished {}'.format(input_path))
    except:
        print('Error with {}: {}'.format(input_path, traceback.format_exc()))

def normalize_paths(base_path, paths):

    new_paths = []
    for name in paths:
        path = base_path + name
        cubes_paths = glob(path + '/*/')
        cubes_paths_normalized = ['/' + name + '/' + os.path.normpath(path_iter).split(os.sep)[-1] for path_iter in
                                  cubes_paths]

        new_paths = new_paths + cubes_paths_normalized

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--res', type=int)
    parser.add_argument("--input_path", type = str)
    parser.add_argument("--output_path", type = str , help = "without hyphen at the end")
    parser.add_argument("--sigmas", nargs = '+', type = float)
    parser.add_argument("--density", type = float, default = 0.001708246)
    parser.add_argument("--split_file", type = str, default = "../datasets/split_scenes_names.npz" )
    args = parser.parse_args()

    paths = glob(args.input_path + '/*/')
    paths.sort()
    paths = paths[:100]
    random.shuffle(paths)

    p = Pool(mp.cpu_count())
    p.map(partial(bd_sampl_vx_ptcld, sigmas=args.sigmas, res = args.res, density = args.density, output_path = args.output_path), paths)
    p.close()
    p.join()

    data = np.load(args.split_file)
    modes = ['train', 'test', 'val']
    new_dict = {}
    for mode in modes:
        new_dict[mode] = normalize_paths(base_path=args.output_path, paths =data[mode])

    np.savez('../datasets/split_scenes.npz', train = new_dict['train'], test = new_dict['test'], val = new_dict['val'])






