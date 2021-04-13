# Main SMPLpix Training Script
#
# (c) Sergey Prokudin (sergey.prokudin@gmail.com), 2021
#

import os
import pprint
from torch.utils.data import DataLoader

from smplpix.args import get_smplpix_arguments
from smplpix.utils import generate_mp4
from smplpix.dataset import SMPLPixDataset
from smplpix.unet import UNet
from smplpix.training import train, evaluate


def generate_eval_video(args, data_dir, unet, frame_rate=25, save_target=False, save_input=True):

    # we will now use the network trained on 20 sketches to convert the rest of AMASS renders
    print("rendering SMPLpix predictions for %s..." % data_dir)
    test_dataset = SMPLPixDataset(data_dir=data_dir,
                                  downsample_factor=args.downsample_factor,
                                  perform_augmentation=False,
                                  n_input_channels=args.n_input_channels,
                                  n_output_channels=args.n_output_channels,
                                  augmentation_probability=args.aug_prob)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    final_renders_path = os.path.join(args.workdir, 'final_test_renders')
    _ = evaluate(unet, test_dataloader, final_renders_path, args.device, save_target=save_target, save_input=save_input)

    print("generating video animation for data %s..." % data_dir)
    data_dir_last = os.path.split(data_dir)[-1]

    video_animation_path = os.path.join(args.workdir, '%s_animation.mp4' % data_dir_last)
    _ = generate_mp4(final_renders_path, video_animation_path, frame_rate=frame_rate)

    return

def main():

    print("******************************************************************************************\n"+
          "****************************** SMPLpix Training Loop  ************************************\n"+
          "******************************************************************************************\n")

    args = get_smplpix_arguments()
    print("ARGS:")
    pprint.pprint(args)

    log_dir = os.path.join(args.workdir, 'logs')
    ckpt_path = os.path.join(args.workdir, 'network.h5')

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

    if args.val_dir is not None:
        val_dataset = SMPLPixDataset(data_dir=val_dir,
                                       perform_augmentation=True,
                                       augmentation_probability=args.aug_prob,
                                       downsample_factor=args.downsample_factor,
                                       n_input_channels=args.n_input_channels,
                                       n_output_channels=args.n_output_channels)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    else:
        print("no validation data was provided, will use train data for validation...")
        val_dataloader = train_dataloader

    print("defining the neural renderer model (U-Net)...")
    unet = UNet(in_channels=args.n_input_channels, out_channels=args.n_output_channels,
                n_blocks=args.n_unet_blocks, dim=2, up_mode='resizeconv_linear').to(args.device)

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
        generate_eval_video(args, test_dir, unet, save_target=False)

    if finished:
        generate_eval_video(args, train_dir, unet, save_target=True)
        generate_eval_video(args, val_dir, unet, save_target=True)
        generate_eval_video(args, test_dir, unet, save_target=False)

    return

if __name__== '__main__':
    main()
