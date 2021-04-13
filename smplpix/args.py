import os
import argparse

def get_smplpix_arguments():

    parser = argparse.ArgumentParser(description='SMPLpix argument parser')
    parser.add_argument('--workdir',
                        dest='workdir',
                        help='workdir to save data, checkpoints, renders, etc.',
                        default=os.getcwd())
    parser.add_argument('--train_dir',
                        dest='train_dir',
                        help='directory with training input and target images to the network, should contain'
                             'input and output subfolders',
                        default=None)
    parser.add_argument('--val_dir',
                        dest='val_dir',
                        help='directory with validation set',
                        default=None)
    parser.add_argument('--test_dir',
                        dest='test_dir',
                        help='directory with test set (might only include input folder)',
                        default=None)
    parser.add_argument('--n_input_channels',
                        dest='n_input_channels',
                        type=int,
                        help='number of channels in the input images',
                        default=3)
    parser.add_argument('--n_output_channels',
                        dest='n_output_channels',
                        type=int,
                        help='number of channels in the input images',
                        default=3)
    parser.add_argument('--n_unet_blocks',
                        dest='n_unet_blocks',
                        type=int,
                        help='number of blocks in UNet rendering module',
                        default=5)
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
    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        type=float,
                        help='initial learning rate',
                        default=1.0e-4)
    parser.add_argument('--eval_every_nth_epoch',
                        dest='eval_every_nth_epoch',
                        type=int,
                        help='evaluate on validation data every nth epoch',
                        default=25)
    parser.add_argument('--sched_patience',
                        dest='sched_patience',
                        type=int,
                        help='amount of validation set evaluations with no improvement after which LR will be reduced',
                        default=5)
    parser.add_argument('--aug_prob',
                        dest='aug_prob',
                        type=float,
                        help='amount of validation set evaluations with no improvement after which LR will be reduced',
                        default=0.25)

    args = parser.parse_args()

    return args