from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import traceback


class VoxelizedDataset(Dataset):


    def __init__(self, mode,  data_path, split_file, res = 256, density =0,
                 pointcloud_samples = 3000,  batch_size = 1, num_sample_points = 1024, num_workers = 1,
                 sample_distribution = [1], sample_sigmas = [0.01], **kwargs):
        """
        :param mode: train|test|val
        :param data_path: path where data is stored
        :param split_file: location of split file
        :param res: resolution of input voxelized point cloud
        :param density: Density used for generating input and boundary point clouds (one of density or pointcloud_samples must be specified)
        :param pointcloud_samples: Number of samples used for generating input point cloud (one of density or pointcloud_samples must be specified)
        :param batch_size: batch size
        :param num_sample_points: total points used as input for the neural network
        :param num_workers: Num workers used for training
        :param sample_distribution: What fraction to use from boundary points generated using different sigmas
        :param sample_sigmas: Sigmas used for boundary points generation
        :param kwargs:
        """
        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        self.density = density
        self.res = res

        self.data = np.load(split_file)[mode]

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pointcloud_samples = pointcloud_samples

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.path + self.data[idx]

        try:
            if self.density:
                voxel_path = path + '/voxelized_point_cloud_{}res_{}density.npz'.format(self.res, self.pointcloud_samples)
            if self.pointcloud_samples:
                voxel_path = path + '/voxelized_point_cloud_{}res_{}points.npz'.format(self.res, self.pointcloud_samples)

            occupancies = np.unpackbits(np.load(voxel_path)['compressed_occupancies'])
            input = np.reshape(occupancies, (self.res,)*3)

            points = []
            coords = []
            df = []

            for i, num in enumerate(self.num_samples):
                boundary_samples_path = path + '/{}boundary_{}_samples.npz'.format('pymesh_', self.sample_sigmas[i])
                boundary_samples_npz = np.load(boundary_samples_path)
                boundary_sample_points = boundary_samples_npz['points']
                boundary_sample_coords = boundary_samples_npz['grid_coords']
                boundary_sample_df = boundary_samples_npz['df']
                subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
                points.extend(boundary_sample_points[subsample_indices])
                coords.extend(boundary_sample_coords[subsample_indices])
                df.extend(boundary_sample_df[subsample_indices])

            assert len(points) == self.num_sample_points
            assert len(df) == self.num_sample_points
            assert len(coords) == self.num_sample_points
        except:
            print('Error with {}: {}'.format(path, traceback.format_exc()))
            raise

        return {'grid_coords':np.array(coords, dtype=np.float32),'df': np.array(df, dtype=np.float32),'points':np.array(points, dtype=np.float32), 'inputs': np.array(input, dtype=np.float32), 'path' : path}

    def get_loader(self, shuffle =True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)


