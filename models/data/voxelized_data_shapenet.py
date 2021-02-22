from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import traceback


class VoxelizedDataset(Dataset):


    def __init__(self, mode, res, pointcloud_samples, data_path, split_file ,
                 batch_size, num_sample_points, num_workers, sample_distribution, sample_sigmas):

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        self.split = np.load(split_file)

        self.mode = mode
        self.data = self.split[mode]
        self.res = res

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pointcloud_samples = pointcloud_samples

        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:

            path = self.data[idx]
            input_path = path
            samples_path = path

            voxel_path = input_path + '/voxelized_point_cloud_{}res_{}points.npz'.format(self.res, self.pointcloud_samples)
            occupancies = np.unpackbits(np.load(voxel_path)['compressed_occupancies'])
            input = np.reshape(occupancies, (self.res,)*3)

            if self.mode == 'test':
                return {'inputs': np.array(input, dtype=np.float32), 'path' : path}

            points = []
            coords = []
            df = []

            for i, num in enumerate(self.num_samples):
                boundary_samples_path = samples_path + '/boundary_{}_samples.npz'.format( self.sample_sigmas[i])
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
