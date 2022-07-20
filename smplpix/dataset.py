# SMPLpix basic dataset class used in all experiments
#
# (c) Sergey Prokudin (sergey.prokudin@gmail.com), 2021
#

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvf

class SMPLPixDataset(Dataset):

    def __init__(self, data_dir,
                 n_input_channels=3,
                 n_output_channels=3,
                 downsample_factor=1,
                 perform_augmentation=False,
                 augmentation_probability=0.75,
                 aug_scale_interval=None,
                 aug_angle_interval=None,
                 aug_translate_interval=None,
                 input_fill_color=1,
                 output_fill_color=1):

        if aug_translate_interval is None:
            aug_translate_interval = [-100, 100]
        if aug_scale_interval is None:
            aug_scale_interval = [0.5, 1.5]
        if aug_angle_interval is None:
            aug_angle_interval = [-60, 60]

        self.input_dir = os.path.join(data_dir, 'input')
        self.output_dir = os.path.join(data_dir, 'output')
        if not os.path.exists(self.output_dir):
            self.output_dir = self.input_dir

        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.samples = sorted(os.listdir(self.output_dir))
        self.downsample_factor = downsample_factor
        self.perform_augmentation = perform_augmentation
        self.augmentation_probability = augmentation_probability
        self.aug_scale_interval = aug_scale_interval
        self.aug_angle_interval = aug_angle_interval
        self.aug_translate_interval = aug_translate_interval
        self.input_fill_color = input_fill_color
        self.output_fill_color = output_fill_color

    def __len__(self):
        return len(self.samples)

    def _get_augmentation_params(self):

        scale = np.random.uniform(low=self.aug_scale_interval[0],
                                  high=self.aug_scale_interval[1])
        angle = np.random.uniform(self.aug_angle_interval[0], self.aug_angle_interval[1])
        translate = [np.random.uniform(self.aug_translate_interval[0],
                                       self.aug_translate_interval[1]),
                     np.random.uniform(self.aug_translate_interval[0],
                                       self.aug_translate_interval[1])]

        return scale, angle, translate

    def _augment_images(self, x, y):

        augment_instance = np.random.uniform() < self.augmentation_probability

        if augment_instance:
            scale, angle, translate = self._get_augmentation_params()

            x = tvf.affine(x,
                           angle=angle,
                           translate=translate,
                           scale=scale,
                           shear=0, fill=self.input_fill_color)

            y = tvf.affine(y,
                           angle=angle,
                           translate=translate,
                           scale=scale,
                           shear=0, fill=self.output_fill_color)

        return x, y

    def __getitem__(self, idx):

        img_name = self.samples[idx]
        x_path = os.path.join(self.input_dir, img_name)
        x = Image.open(x_path)
        y_path = os.path.join(self.output_dir, img_name)
        y = Image.open(y_path)

        if self.perform_augmentation:
            x, y = self._augment_images(x, y)

        x = torch.Tensor(np.asarray(x) / 255).transpose(0, 2)
        y = torch.Tensor(np.asarray(y) / 255).transpose(0, 2)
        x = x[0:self.n_input_channels, ::self.downsample_factor, ::self.downsample_factor]
        y = y[0:self.n_output_channels, ::self.downsample_factor, ::self.downsample_factor]

        return x, y, img_name
