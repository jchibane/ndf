import models.local_model as model
from models.data import dataloader_garments, voxelized_data_shapenet

from models import generation
import torch
from torch.nn import functional as F


def rot_YZ(points):
    points_rot = points.copy()
    points_rot[:, 1], points_rot[:, 2] = points[:, 2], points[:, 1]
    return points_rot

def to_grid(points):
    grid_points = points.copy()
    grid_points[:, 0], grid_points[:, 2] = points[:, 2], points[:, 0]

    return 2 * grid_points

def from_grid(grid_points):
    points = grid_points.copy()
    points[:, 0], points[:, 2] = grid_points[:, 2], grid_points[:, 0]

    return 0.5 * points

# 'test', 'val', 'train'
def loadNDF(index, pointcloud_samples, exp_name, data_dir, split_file, sample_distribution, sample_sigmas, res,  mode = 'test'):

    global encoding
    global net
    global device

    net = model.NDF()

    device = torch.device("cuda")


    if 'garments' in exp_name.lower() :

        dataset = dataloader_garments.VoxelizedDataset(mode =  mode,  data_path = data_dir, split_file = split_file,
                                                        res = res, density =0, pointcloud_samples = pointcloud_samples,
                                                       sample_distribution=sample_distribution,
                                                       sample_sigmas=sample_sigmas,
                                                        )



        checkpoint = 'checkpoint_127h:6m:33s_457593.9149734974'

        generator = generation.Generator(net,exp_name, checkpoint = checkpoint, device = device)

    if 'cars' in exp_name.lower() :

        dataset = voxelized_data_shapenet.VoxelizedDataset( mode = mode, res = res, pointcloud_samples =  pointcloud_samples,
                                                   data_path = data_dir, split_file = split_file,
                                                   sample_distribution = sample_distribution, sample_sigmas = sample_sigmas,
                                                   batch_size = 1, num_sample_points = 1024, num_workers = 1
                                                   )



        checkpoint = 'checkpoint_108h:5m:50s_389150.3971107006'

        generator = generation.Generator(net, exp_name, checkpoint=checkpoint, device=device)


    example = dataset[index]

    print('Object: ',example['path'])
    inputs = torch.from_numpy(example['inputs']).unsqueeze(0).to(device) # lead inputs and samples including one batch channel

    for param in net.parameters():
        param.requires_grad = False

    encoding = net.encoder(inputs)



def predictRotNDF(points):

    points = rot_YZ(points)
    points = to_grid(points)
    points = torch.from_numpy(points).unsqueeze(0).float().to(device)
    return torch.clamp(net.decoder(points,*encoding), max=0.1).squeeze(0).cpu().numpy()


def predictRotGradientNDF(points):
    points = rot_YZ(points)
    points = to_grid(points)
    points = torch.from_numpy(points).unsqueeze(0).float().to(device)
    points.requires_grad = True

    df_pred = torch.clamp(net.decoder(points,*encoding), max=0.1)

    df_pred.sum().backward()

    gradient = F.normalize(points.grad, dim=2)[0].detach().cpu().numpy()

    df_pred = df_pred.detach().squeeze(0).cpu().numpy()
    return df_pred, rot_YZ( 2 * from_grid(gradient))
