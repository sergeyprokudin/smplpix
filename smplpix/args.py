import os
import argparse

def get_smplpix_arguments():

    parser = argparse.ArgumentParser(description='SMPLpix argument parser')
    parser.add_argument('--workdir',
                        dest='workdir',
                        help='workdir to save data, checkpoints, renders, etc.',
                        default=os.getcwd())
    parser.add_argument('--input_dir',
                        dest='input_dir',
                        help='directory with input images to the network (e.g. point cloud projections, '
                             'coarse mesh renders, etc.)',
                        default=None)
    parser.add_argument('--output_dir',
                        dest='output_dir',
                        help='directory with corresponding target images to the network (e.g. real photos'
                             'with the same camera parameters)',
                        default=None)
    parser.add_argument('--n_input_channels',
                        dest='n_input_channels',
                        type=int,
                        help='number of channels in the input images',
                        default=3)
    parser.add_argument('--n_output_channels',
                        dest='n_input_channels',
                        type=int,
                        help='number of channels in the input images',
                        default=3)
    parser.add_argument('--batch_size',
                        dest='batch_size',
                        type=int,
                        help='batch size to use during training',
                        default=4)
    parser.add_argument('--device',
                        dest='device',
                        help='GPU device to use during training',
                        default='cuda')
    parser.add_argument('--downsample_factor',
                        dest='downsample_factor',
                        type=int,
                        help='image downsampling factor (for faster training)',
                        default=4)
    parser.add_argument('--n_epochs',
                        dest='n_epochs',
                        type=int,
                        help='number of epochs to train the network for',
                        default=500)
    parser.add_argument('--eval_every_nth_epoch',
                        dest='eval_every_nth_epoch',
                        type=int,
                        help='evaluate on validation data every nth epoch',
                        default=25)
    parser.add_argument('--sched_patience',
                        dest='sched_patience',
                        type=int,
                        help='amount of validation evaluations with no improvement after which LR will be reduced',
                        default=5)
    args = parser.parse_args()

    return args