# Main SMPLpix Training Script
#
# (c) Sergey Prokudin (sergey.prokudin@gmail.com), 2021
#

import os
import shutil
import pprint
import torch
from torch.utils.data import DataLoader

from smplpix.args import get_smplpix_arguments
from smplpix.utils import generate_mp4
from smplpix.dataset import SMPLPixDataset
from smplpix.unet import UNet
from smplpix.training import train, evaluate
from smplpix.utils import download_and_unzip

def generate_eval_video(args, data_dir, unet, frame_rate=25, save_target=False, save_input=True):

    print("rendering SMPLpix predictions for %s..." % data_dir)
    data_part_name = os.path.split(data_dir)[-1]

    test_dataset = SMPLPixDataset(data_dir=data_dir,
                                  downsample_factor=args.downsample_factor,
                                  perform_augmentation=False,
                                  n_input_channels=args.n_input_channels,
                                  n_output_channels=args.n_output_channels,
                                  augmentation_probability=args.aug_prob)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    final_renders_path = os.path.join(args.workdir, 'renders_%s' % data_part_name)
    _ = evaluate(unet, test_dataloader, final_renders_path, args.device, save_target=save_target, save_input=save_input)

    print("generating video animation for data %s..." % data_dir)

    video_animation_path = os.path.join(args.workdir, '%s_animation.mp4' % data_part_name)
    _ = generate_mp4(final_renders_path, video_animation_path, frame_rate=frame_rate)
    print("saved animation video to %s" % video_animation_path)

    return

def main():

    print("******************************************************************************************\n"+
          "****************************** SMPLpix Training Loop  ************************************\n"+
          "******************************************************************************************\n"+
          "******** Copyright (c) 2021 - now, Sergey Prokudin (sergey.prokudin@gmail.com) ***********\n"+
          "****************************************************************************************+*\n\n")

    args = get_smplpix_arguments()
    print("ARGUMENTS:")
    pprint.pprint(args)

    if not os.path.exists(args.workdir):
        os.makedirs(args.workdir)
    
    log_dir = os.path.join(args.workdir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    if args.data_url is not None:
        download_and_unzip(args.data_url, args.workdir)
        args.data_dir = os.path.join(args.workdir, 'smplpix_data')

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'validation')
    test_dir = os.path.join(args.data_dir, 'test')

    train_dataset = SMPLPixDataset(data_dir=train_dir,
                             perform_augmentation=True,
                             augmentation_probability=args.aug_prob,
                             downsample_factor=args.downsample_factor,
                             n_input_channels=args.n_input_channels,
                             n_output_channels=args.n_output_channels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    if os.path.exists(val_dir):
        val_dataset = SMPLPixDataset(data_dir=val_dir,
                                       perform_augmentation=False,
                                       augmentation_probability=args.aug_prob,
                                       downsample_factor=args.downsample_factor,
                                       n_input_channels=args.n_input_channels,
                                       n_output_channels=args.n_output_channels)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    else:
        print("no validation data was provided, will use train data for validation...")
        val_dataloader = train_dataloader

    print("defining the neural renderer model (U-Net)...")
    unet = UNet(in_channels=args.n_input_channels, 
                out_channels=args.n_output_channels,
                sigmoid_output=args.sigmoid_output,
                n_blocks=args.n_unet_blocks, dim=2, up_mode='resizeconv_linear').to(args.device)

    if args.checkpoint_path is None:
        ckpt_path = os.path.join(args.workdir, 'network.h5')
    else:
        ckpt_path = args.checkpoint_path
    
    if args.resume_training and os.path.exists(ckpt_path):
        print("found checkpoint, resuming from: %s" % ckpt_path)
        unet.load_state_dict(torch.load(ckpt_path))
    if not args.resume_training:
        print("starting training from scratch, cleaning the log dirs...")
        shutil.rmtree(log_dir)
    
    print("starting training...")
    finished = False
    try:
        train(model=unet, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
              log_dir=log_dir, ckpt_path=ckpt_path, device=args.device, n_epochs=args.n_epochs,
              eval_every_nth_epoch=args.eval_every_nth_epoch, sched_patience=args.sched_patience,
              init_lr=args.learning_rate)
        finished = True

    except KeyboardInterrupt:
        print("training interrupted, generating final animations...")
        generate_eval_video(args, train_dir, unet, save_target=True)
        generate_eval_video(args, val_dir, unet, save_target=True)
        generate_eval_video(args, test_dir, unet, save_target=True)

    if finished:
        generate_eval_video(args, train_dir, unet, save_target=True)
        generate_eval_video(args, val_dir, unet, save_target=True)
        generate_eval_video(args, test_dir, unet, save_target=True)

    return

if __name__== '__main__':
    main()
