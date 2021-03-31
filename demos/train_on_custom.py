# Training SMPLpix on hand-drawn sketches created on top of mesh renders
#
# Training data:
#  - input: AMASS renders (https://amass.is.tue.mpg.de/) of a few CMU sequences (http://mocap.cs.cmu.edu/);
#  - target: 20 hand-drawn sketches created by an artist Alexander Kabarov (blackocher@gmail.com)
#
# (c) Sergey Prokudin (sergey.prokudin@gmail.com), 2021
#

import os
from torch.utils.data import DataLoader

from smplpix.args import get_smplpix_arguments
from smplpix.utils import get_amass_cmu_sketch_data, generate_mp4
from smplpix.dataset import SMPLPixDataset
from smplpix.unet import UNet
from smplpix.training import train, evaluate


def main():

    print("******************************************************************************************\n"+
          "********* Training SMPLpix on hand-drawn sketches created on top of mesh renders *********\n"+
          "******************************************************************************************\n")

    args = get_smplpix_arguments()
    log_dir = os.path.join(args.workdir, 'logs')
    ckpt_path = os.path.join(args.workdir, 'network.h5')

    dataset = SMPLPixDataset(input_dir=args.input_dir,
                             output_dir=args.output_dir,
                             perform_augmentation=True,
                             downsample_factor=args.downsample_factor,
                             n_input_channels=args.n_input_channels,
                             n_output_channels=args.n_output_channels)

    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    print("defining the neural renderer model (U-Net)...")
    unet = UNet(n_channels=args.n_input_channels, n_classes=args.n_output_channels).to(args.device)

    print("starting training...")
    train(model=unet, train_dataloader=dataloader, val_dataloader=dataloader,
          log_dir=log_dir, ckpt_path=ckpt_path, device=args.device, n_epochs=args.n_epochs,
          eval_every_nth_epoch=args.eval_every_nth_epoch, sched_patience=args.sched_patience,
          init_lr=args.learning_rate)

    # we will now use the network trained on 20 sketches to convert the rest of AMASS renders
    print("processing test AMASS renders...")
    test_dataset = SMPLPixDataset(input_dir=args.input_dir,
                                  downsample_factor=args.downsample_factor,
                                  perform_augmentation=False,
                                  n_input_channels=args.n_input_channels,
                                  n_output_channels=args.n_output_channels)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    final_renders_path = os.path.join(args.workdir, 'final_test_renders')
    _ = evaluate(unet, test_dataloader, final_renders_path, args.device, save_target=False, save_input=True)

    print("generating video animation...")
    video_animation_path = os.path.join(args.workdir, 'animation.mp4')
    _ = generate_mp4(final_renders_path, video_animation_path, img_ext='png', frame_rate=15)

    print("all done.")
    print("network checkpoint: %s" % ckpt_path)
    print("generated renders: %s" % final_renders_path)
    print("generated animation video: %s" % video_animation_path)

    return

if __name__== '__main__':
    main()
