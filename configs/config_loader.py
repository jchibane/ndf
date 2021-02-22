import configargparse
import numpy as np
import os


def config_parser():
    parser = configargparse.ArgumentParser()

    # Experiment Setup
    parser.add_argument('--config', is_config_file=True, default='configs/shapenet_cars.txt',
                        help='config file path')
    parser.add_argument("--exp_name", type=str, default=None,
                        help='Experiment name, used as folder name for the experiment. If left blank, a \
                         name will be auto generated based on the configuration settings.')
    parser.add_argument("--data_dir", type=str,
                        help='input data directory')
    parser.add_argument("--input_data_glob", type=str,
                        help='glob expression to find raw input files')
    parser.add_argument("--split_file", type=str,
                        help='Path to read and write the data split file. Needs to end with ".npz"')

    # Training Data Parameters
    parser.add_argument("--sample_std_dev", action='append', type=float,
                        help='Standard deviations of gaussian samples. \
                Used for displacing surface points to sample the distance field.')
    parser.add_argument("--sample_ratio", action='append', type=float,
                        help='Ratio of standard deviations for samples used for training. \
                Needs to have the same len as sample_std with floats between 0-1 \
                and summing to 1.')
    parser.add_argument("--bb_min", default=-0.5, type=float,
                        help='Training and testing shapes are normalized to be in a common bounding box.\
                             This value defines the min value in x,y and z for the bounding box.')
    parser.add_argument("--bb_max", default=0.5, type=float,
                        help='Training and testing shapes are normalized to be in a common bounding box.\
                             This value defines the max value in x,y and z for the bounding box.')
    parser.add_argument("--input_res", type=int, default=256,
                        help='Training and testing shapes are normalized to be in a common bounding box.\
                             This value defines the max value in x,y and z for the bounding box.')
    parser.add_argument("--num_points", type=int, default=10000,
                        help='Number of points sampled from each ground truth shape.')

    # Preprocessing - Multiprocessing
    parser.add_argument("--num_chunks", type=int, default=1,
                        help='The preprocessing can be distributed over num_chunks multiple machines.\
                         For this the raw files are split into num_chunks chunks. \
                        Default is preprocessing on a single machine.')
    parser.add_argument("--current_chunk", type=int, default=0,
                        help='Tells the script which chunk it should process. \
                Value between 0 till num_chunks-1.')
    parser.add_argument("--num_cpus", type=int, default=-1,
                        help='Number of cpu cores to use for running the script. \
            Default is -1, that is, using all available cpus.')

    # Creating a data test/train/validation split
    parser.add_argument('--class_folders', type=str, default=None,
                       help='If set to None, the split is created by creating a random sample from all input files. '
                            'If not None, the split is created per class of objects. Objects of the same class need to '
                            'be in a common parent folder for this. Variable class_folder is interpreted as glob '
                            'pattern, suffix of data_dir - i.e. data_dir + class_folder, e.g. class_folder="/*/".')

    parser_nval = parser.add_mutually_exclusive_group()
    parser_nval.add_argument('--n_val', type=int,
                             help='Size of validation set.')
    parser_nval.add_argument('--r_val', type=float, default=0.1,
                             help='Relative size of validation set.')

    parser_ntest = parser.add_mutually_exclusive_group()
    parser_ntest.add_argument('--n_test', type=int,
                              help='Size of test set.')
    parser_ntest.add_argument('--r_test', type=float, default=0.2,
                              help='Relative size of test set.')

    # Generation
    parser.add_argument("--num_sample_points_generation", type=int, default=50000,
                        help='Number of point samples per object provided to the NDF network during generation.\
                            Influences generation speed (larger batches result in faster generation) but also GPU \
                             memory usage (higher values need more memory). Tip: choose largest possible value on GPU.')

    # Training
    parser.add_argument("--num_sample_points_training", type=int, default=50000,
                        help='Number of point samples per object provided to the NDF network during training.\
                            Influences training speed (larger batches result in shorter epochs) but also GPU \
                             memory usage (higher values need more memory). Needs to be balanced with batch_size.')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='Number of objects provided to the NDF network in one batch during training.\
                            Influences training speed (larger batches result in shorter epochs) but also GPU \
                             memory usage (higher values need more memory). Needs to be balanced with \
                             num_sample_points_training')
    parser.add_argument("--num_epochs", type=int, default=1500,
                        help='Stopping citron for duration of training. Model converges much earlier: model convergence\
                         can be checked via tensorboard and is logged within the experiment folder.')
    parser.add_argument("--lr", type=float, default=1e-6,
                        help='Learning rate used during training.')
    parser.add_argument("--optimizer", type=str, default='Adam',
                        help='Optimizer used during training.')


    return parser


def get_config():
    parser = config_parser()
    cfg = parser.parse_args()

    cfg.sample_ratio = np.array(cfg.sample_ratio)
    cfg.sample_std_dev = np.array(cfg.sample_std_dev)

    assert np.sum(cfg.sample_ratio) == 1
    assert np.any(cfg.sample_ratio < 0) == False
    assert len(cfg.sample_ratio) == len(cfg.sample_std_dev)

    if cfg.exp_name is None:
        cfg.exp_name = 'data-{}dist-{}sigmas-{}res-{}'.format(
                                                        os.path.basename(cfg.data_dir),
                                                        ''.join(str(e) + '_' for e in cfg.sample_ratio),
                                                        ''.join(str(e) + '_' for e in cfg.sample_std_dev),
                                                        cfg.input_res)

    return cfg
