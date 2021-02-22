import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models import training
import torch
import configs.config_loader as cfg_loader


cfg = cfg_loader.get_config()
net = model.NDF()



train_dataset = voxelized_data.VoxelizedDataset('train',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=cfg.split_file,
                                          batch_size=cfg.batch_size,
                                          num_sample_points=cfg.num_sample_points_training,
                                          num_workers=30,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)
val_dataset = voxelized_data.VoxelizedDataset('val',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=cfg.split_file,
                                          batch_size=cfg.batch_size,
                                          num_sample_points=cfg.num_sample_points_training,
                                          num_workers=30,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)



trainer = training.Trainer(net,
                           torch.device("cuda"),
                           train_dataset,
                           val_dataset,
                           cfg.exp_name,
                           optimizer=cfg.optimizer,
                           lr=cfg.lr)

trainer.train_model(cfg.num_epochs)
