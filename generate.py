import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
import numpy as np
from models.generation import Generator
from generation_iterator import gen_iterator
import torch
import argparse


parser = argparse.ArgumentParser(
    description='Run Training'
)

parser.add_argument('-pc_samples' , default=10000, type=int)
parser.add_argument('-dist','--sample_distribution', default=[0.01, 0.49, 0.5], nargs='+', type=float)
parser.add_argument('-std_dev','--sample_sigmas',default=[0.08, 0.02, 0.003], nargs='+', type=float)
parser.add_argument('-res' , default=256, type=int)
parser.add_argument('-pretrained', dest='pretrained', action='store_true')
args = parser.parse_args()


split_file = 'shapenet/split_cars.npz'
model_name = 'SVR'
device = torch.device("cuda")
net = model.SVR_enc_dec()


dataset = voxelized_data.VoxelizedDataset('test', pointcloud_samples= args.pc_samples, res=args.res, sample_distribution=args.sample_distribution,
                                          sample_sigmas=args.sample_sigmas ,num_sample_points=50000, batch_size=1, num_workers=30, split_file = split_file)


exp_name = '{}dist-{}sigmas-{}res-{}'.format( 'pretrained_' if args.pretrained else '',
                                        ''.join(str(e)+'_' for e in args.sample_distribution),
                                       ''.join(str(e) +'_'for e in args.sample_sigmas),
                                                                args.res)


gen = Generator(net,exp_name, device = device)

out_path = 'experiments/{}/evaluation/'.format(exp_name)

gen_iterator(out_path, dataset, gen)
