# Training SMPLpix on hand-drawn sketches created on top of mesh renders
#
# Training data:
#  - input: AMASS renders (https://amass.is.tue.mpg.de/) of a few CMU sequences (http://mocap.cs.cmu.edu/);
#  - target: 20 hand-drawn sketches created by an artist Alexander Kabarov (blackocher@gmail.com)
#
# (c) Sergey Prokudin (sergey.prokudin@gmail.com), 2021
#

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from smplpix.unet import UNet
from smplpix.vgg import Vgg16
from smplpix.dataset import SMPLPixDataset
from smplpix.utils import get_amass_cmu_sketch_data
from smplpix.training import train, evaluate

def get_args():

    parser = argparse.ArgumentParser(description='Train SMPLpix rendering network on 20 hand-drawn sketches.')
    parser.add_argument('--workdir',
                        dest='workdir',
                        help='workdir to save data, checkpoints, renders, etc.',
                        default=os.getcwd())
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
    print(vars(args))

    return args

def main():

    print("*********Training SMPLpix on hand-drawn sketches created on top of mesh renders*********")
    args = get_args()
    log_dir = os.path.join(args.workdir, 'logs')
    ckpt_path = os.path.join(args.workdir, 'kabarov_net.h5')
    renders_dir, sketch_dir = get_amass_cmu_sketch_data(args.workdir)

    dataset = SMPLPixDataset(renders_dir,
                             sketch_dir,
                             perform_augmentation=True,
                             downsample_factor=args.downsample_factor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    unet = UNet(n_channels=3, n_classes=3).to(args.device)

    train(model=unet, train_dataloader=dataloader, val_dataloader=dataloader,
          log_dir=log_dir, ckpt_path=ckpt_path, device=args.device, n_epochs=args.n_epochs,
          eval_every_nth_epoch=args.eval_every_nth_epoch, sched_patience=args.sched_patience)

    # we will now use the network trained on 20 sketches to convert the rest of AMASS renders
    test_dataset = SMPLPixDataset(renders_dir, renders_dir,
                                  downsample_factor=args.downsample_factor,
                                  perform_augmentation=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    final_renders_path = os.path.join(args.workdir, 'final_test_renders')
    _ = evaluate(unet, test_dataloader, final_renders_path, args.device)

    print("all done.")
    print("network checkpoint: %s" % ckpt_path)
    print("generated renders: %s")
    #print("generated animation video: %s" % )

    return

if __name__== '__main__':
    main()
