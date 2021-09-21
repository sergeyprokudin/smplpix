import os
import argparse

def get_smplpix_arguments():

    parser = argparse.ArgumentParser(description='SMPLpix argument parser')
    parser.add_argument('--workdir',
                        dest='workdir',
                        help='workdir to save data, checkpoints, renders, etc.',
                        default=os.getcwd())
    parser.add_argument('--data_dir',
                        dest='data_dir',
                        help='directory with training input and target images to the network, should contain'
                             'input and output subfolders',
                        default=None)
    parser.add_argument('--resume_training',
                        dest='resume_training',
                        type=int,
                        help='whether to continue training process given the checkpoint in workdir',
                        default=False)
    parser.add_argument('--data_url',
                        dest='data_url',
                        help='Dropbox URL containing zipped dataset',
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
    parser.add_argument('--sigmoid_output',
                    dest='sigmoid_output',
                    type=int,
                    help='whether to add sigmoid activation as a final layer',
                    default=True)
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
                        default=1.0e-3)
    parser.add_argument('--eval_every_nth_epoch',
                        dest='eval_every_nth_epoch',
                        type=int,
                        help='evaluate on validation data every nth epoch',
                        default=10)
    parser.add_argument('--sched_patience',
                        dest='sched_patience',
                        type=int,
                        help='amount of validation set evaluations with no improvement after which LR will be reduced',
                        default=3)
    parser.add_argument('--aug_prob',
                        dest='aug_prob',
                        type=float,
                        help='probability that the input sample will be rotated and rescaled - higher value is recommended for data scarse scenarios',
                        default=0.8)
    parser.add_argument('--save_target',
                        dest='save_target',
                        type=int,
                        help='whether to save target images during evaluation',
                        default=1)
    parser.add_argument('--checkpoint_path',
                        dest='checkpoint_path',
                        help='path to checkpoint (for evaluation)',
                        default=None)
    args = parser.parse_args()

    return args
