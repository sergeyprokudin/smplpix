https://user-images.githubusercontent.com/8117267/116540639-b9537480-a8ea-11eb-81ca-57d473147fbd.mp4


_**Left**: [SMPL-X](https://smpl-x.is.tue.mpg.de/) human mesh registered with [SMPLify-X](https://smpl-x.is.tue.mpg.de/), **middle**: SMPLpix render, **right**: ground truth video._


# SMPLpix: Neural Avatars from 3D Human Models

*SMPLpix* neural rendering framework combines deformable 3D models such as [SMPL-X](https://smpl-x.is.tue.mpg.de/)
with the power of image-to-image translation frameworks (aka [pix2pix](https://phillipi.github.io/pix2pix/) models).

Please check our [WACV 2021 paper](https://arxiv.org/abs/2008.06872) or a [5-minute explanatory video](https://www.youtube.com/watch?v=JY9t4xUAouk) for more details on the framework. 

_**Note**_: this repository is a re-implementation of the original framework, made by the same author after the end of internship.
It **does not contain** the original Amazon multi-subject training data and code, and uses full mesh rasterizations as inputs rather than point projections (as described [here](https://youtu.be/JY9t4xUAouk?t=241)).

## Installation

```
pip3 install git+https://github.com/sergeyprokudin/smplpix
```

## Demo

| Description      | Link |
| ----------- | ----------- |
| Process a video into a SMPLpix dataset| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/nerfies/blob/main/notebooks/Nerfies_Capture_Processing.ipynb)|
| Train SMPLpix| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sergeyprokudin/smplpix/blob/main/colab_notebooks/SMPLpix_training.ipynb)|

### Prepare the data

We provide the Colab notebook for preparing SMPLpix training dataset. This will allow you 
to create your own neural avatar given [monocular video of a human moving in front of the camera](https://www.dropbox.com/s/rjqwf894ovso218/smplpix_test_video_na.mp4?dl=0).

### Run demo training

We provide some preprocessed data which allows you to run and test the training pipeline right away:

```
pip3 install git+https://github.com/sergeyprokudin/smplpix
cd smplpix
python setup.py install
python smplpix/train.py --workdir='/content/smplpix_logs/' \
                        --data_url='https://www.dropbox.com/s/coapl05ahqalh09/smplpix_data_test_final.zip?dl=0'
```

### Train on your own data

You can train SMPLpix on your own data by specifying the path to the root directory with data:

```
python smplpix/train.py --workdir='/content/smplpix_logs/' \
                        --data_dir='/path/to/data'
```

The directory should contain train, validation and test folders, each of which should contain input and output folders. Check the structure of [the demo dataset](https://www.dropbox.com/s/coapl05ahqalh09/smplpix_data_test_final.zip?dl=0) for reference.

You can also specify various parameters of training via command line: 

```
python smplpix/train.py --workdir='/content/smplpix_logs/' \
                        --data_dir='/path/to/data' \
                        --downsample_factor=2 \
                        --n_epochs=100 \
                        --n_output_channels==3
```

Check the [args.py](https://github.com/sergeyprokudin/smplpix/blob/main/smplpix/args.py) for the full list of parameters.

## More examples

### Animating with novel poses

https://user-images.githubusercontent.com/8117267/116546566-0edf4f80-a8f2-11eb-9fb2-a173c0018a4e.mp4

**Left**: poses from the test video sequence, **right**: SMPLpix renders. 


### Rendering faces

https://user-images.githubusercontent.com/8117267/116543423-23214d80-a8ee-11eb-9ded-86af17c56549.mp4

_**Left**: [FLAME](https://flame.is.tue.mpg.de/) face model inferred with [DECA](https://github.com/YadiraF/DECA), **middle**: ground truth test video, **right**: SMPLpix render._

Thanks to [Maria Paola Forte](https://www.is.mpg.de/~Forte) for providing the sequence.

### Few-shot artistic neural style transfer

https://user-images.githubusercontent.com/8117267/116544826-e9514680-a8ef-11eb-8682-0ea8d19d0d5e.mp4

_**Left**: rendered [AMASS](https://amass.is.tue.mpg.de/) motion sequence, **right**: generated SMPLpix animations. See [the explanatory video](https://youtu.be/JY9t4xUAouk?t=255) for details._

Credits to [Alexander Kabarov](mailto:blackocher@gmail.com) for providing the training sketches.

## Citation

If you find our work useful in your research, please consider citing:
```
@inproceedings{prokudin2021smplpix,
  title={SMPLpix: Neural Avatars from 3D Human Models},
  author={Prokudin, Sergey and Black, Michael J and Romero, Javier},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1810--1819},
  year={2021}
}
```

## License

See the LICENSE file.

