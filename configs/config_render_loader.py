import configargparse
import numpy as np
import os


def str2bool(inp):
    return inp.lower() in 'true'


def config_parser():
    parser = configargparse.ArgumentParser()

    # Experiment Setup
    parser.add_argument('--config', is_config_file=True, default='configs/garment_render.txt',
                        help='config file path')

    parser.add_argument("--exp_name", type=str, default=None,
                        help='Experiment name, used as folder name for the experiment. If left blank, a \
                         name will be auto generated based on the configuration settings.')

    parser.add_argument("--data_dir", type=str,
                        help='input data directory')

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

    parser.add_argument("--num_points", type=int, default=1000,
                        help='Number of points sampled from each ground truth shape.')

    parser.add_argument("--res", type=int, help='resolution of voxelization')
    parser.add_argument("--pc_samples", type=int, help='input pointcloud size')
    parser.add_argument("--index", type=int, help='index to be rendered')

    ###

    parser.add_argument("--size", type=int, help="the ", default=512)
    parser.add_argument("--max_depth", type=float, help="the ", default=2)
    parser.add_argument("--alpha", type=float, help="the value by which the stepping distance should be multiplied",
                        default=0.6)
    parser.add_argument("--step_back", type=float, default=0.005, help="the ")
    parser.add_argument("--epsilon", type=float, default=0.0026, help="epsilong bal")
    parser.add_argument("--screen_bound", type=float, default=0.4, help="the ")
    parser.add_argument("--screen_depth", type=float, default=-1, help="the ")

    parser.add_argument('--cam_position', nargs='+', type=float, help='3D position of camera', default=[0, 0, -1])
    parser.add_argument('--light_position', nargs='+', type=float, help='3D position of light source',
                        default=[-1, -1, -1])
    parser.add_argument("--cam_orientation", nargs='+', type=float,
                        help="Camera Orientation in xyz euler angles (degrees)", default=[180.0, 0.0, -180.0])

    parser.add_argument("--folder", type=str, default='./save',
                        help="location where images are to be saved")
    parser.add_argument("--shade", type=str2bool, default=True, help="whether to save shade image")
    parser.add_argument("--depth", type=str2bool, default=True, help="whether to save depth image")
    parser.add_argument("--normal", type=str2bool, default=True, help="whether to save normal image")

    parser.add_argument("--debug_mode", type=str2bool, default=True,
                        help="to visualize everything in debug mode or not")

    return parser


def get_config():
    parser = config_parser()
    cfg = parser.parse_args()

    cfg.sample_ratio = np.array(cfg.sample_ratio)
    cfg.sample_std_dev = np.array(cfg.sample_std_dev)

    assert np.sum(cfg.sample_ratio) == 1
    assert np.any(cfg.sample_ratio < 0) == False
    assert len(cfg.sample_ratio) == len(cfg.sample_std_dev)

    return cfg
