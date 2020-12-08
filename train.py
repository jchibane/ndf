import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models import training
import argparse
import torch


parser = argparse.ArgumentParser(
    description='Run Training'
)

parser.add_argument('-pc_samples' , default=10000, type=int)
parser.add_argument('-dist','--sample_distribution', default=[0.01, 0.49, 0.5], nargs='+', type=float)
parser.add_argument('-std_dev','--sample_sigmas',default=[0.08, 0.02, 0.003], nargs='+', type=float)
parser.add_argument('-lr', default=1e-6, type=float)
parser.add_argument('-batch_size' , default=4, type=int)
parser.add_argument('-res' , default=256, type=int)
parser.add_argument('-o','--optimizer' , default='Adam', type=str)

args = parser.parse_args()


net = model.SVR()



split_file = 'shapenet/split_cars.npz'




train_dataset = voxelized_data.VoxelizedDataset('train', pointcloud_samples= args.pc_samples, res=args.res, sample_distribution=args.sample_distribution,
                                          sample_sigmas=args.sample_sigmas ,num_sample_points=50000, batch_size=args.batch_size, num_workers=30, split_file = split_file)

val_dataset = voxelized_data.VoxelizedDataset('val', pointcloud_samples= args.pc_samples, res=args.res, sample_distribution=args.sample_distribution,
                                          sample_sigmas=args.sample_sigmas ,num_sample_points=50000, batch_size=args.batch_size, num_workers=30, split_file = split_file)





exp_name = 'dist-{}sigmas-{}res-{}'.format( ''.join(str(e)+'_' for e in args.sample_distribution),
                                       ''.join(str(e) +'_'for e in args.sample_sigmas),
                                                                args.res)

trainer = training.Trainer(net,torch.device("cuda"),train_dataset, val_dataset,exp_name, optimizer=args.optimizer, lr = args.lr)
trainer.train_model(1500)
