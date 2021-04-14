# SMPLpix: Neural Avatars from Deformable 3D Models

*SMPLpix* neural rendering framework combines two major components: deformable 3D models such as [SMPL-X](https://smpl-x.is.tue.mpg.de/)
with the power of image-to-image translation frameworks (aka [pix2pix](https://phillipi.github.io/pix2pix/) models).

Please check our [WACV 2021 paper](https://arxiv.org/abs/2008.06872) or a [5-minute explanatory video](https://www.youtube.com/watch?v=JY9t4xUAouk) for more details on the framework. 

_**Note**_: this repository is a re-implementation of the original framework using public components (made by the same author after the end of internship).
It **does not contain** the original multi-subject training data and code, and uses full mesh rasterizations as inputs rather than point projections.

## Installation

```
pip3 install git+https://github.com/sergeyprokudin/smplpix
```

## Demo

### Prepare the data

We will provide the script for generating SMPLpix training data from the mp4 video. _Coming soon_.

This uses [SMPLify-X framework](https://smpl-x.is.tue.mpg.de/) for estimating SMPL-X meshes from RGB images.

### Run training

Below is the script

### Render avatar in novel poses and expressions

Once trained, you can render the neural avatar with novel poses and hand\face expressions.

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

This library is licensed under the MIT-0 License. See the LICENSE file. The Model & Software may not be used for pornographic purposes or to generate pornographic material whether commercial or not.

