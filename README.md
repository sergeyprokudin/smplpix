# SMPLpix: Neural Avatars from Deformable 3D Models

*SMPLpix* neural rendering framework combines two major components: deformable 3D models such as [SMPL-X](https://smpl-x.is.tue.mpg.de/)
with the power of image-to-image translation frameworks (aka [pix2pix](https://phillipi.github.io/pix2pix/) models).

Please check our [WACV 2021 paper](https://arxiv.org/abs/2008.06872) or a [5-minute explanatory video](https://www.youtube.com/watch?v=JY9t4xUAouk) for more details on the framework. 

_**Note**_: this repository is a re-implementation of the original framework using public components (made by the same author after the end of internship).
It **does not contain** the original Amazon multi-subject training data and code, and uses full mesh rasterizations as inputs rather than point projections.

## Installation

```
pip3 install git+https://github.com/sergeyprokudin/smplpix
```

## Demo

### Prepare the data

_Coming soon_: we will provide the Colab notebook for preparing SMPLpix training dataset. This will allow you 
to create your own neural avatar given monocular video of a human moving in front of the camera.

### Run training

We provide some preprocessed data which allows you to run and test the training pipeline right away:

```
python smplpix/train.py --data_url='https://www.dropbox.com/s/gcmsf7t1v0snu6i/smplpix_subject0_v2.zip?dl=0' --workdir='./logs' --n_epochs=500 --eval_every_nth_epoch=25  --aug_prob=0.5 --downsample_factor=4  --sched_patience=1
```

### Render avatar in novel poses and expressions

Once trained, you can render the neural avatar with novel poses and hand\face expressions. 

You can download one of the many motion sequences from the AMASS human motion library, and run the following the script

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

This library is licensed under the MIT-0 License. See the LICENSE file. The library may not be used for pornographic purposes or to generate pornographic material whether commercial or not.

