# SMPLpix: Neural Avatars from Deformable 3D Models

*SMPLpix** neural rendering framework combines two major components: deformable 3D models such as [SMPL-x](https://smpl-x.is.tue.mpg.de/)
with the power of image-to-image translation frameworks (aka [pix2pix](https://phillipi.github.io/pix2pix/) models).

Please watch a [5 minute explanatory video](https://www.youtube.com/watch?v=JY9t4xUAouk) for more details on the framework. 

Check our [WACV 2021 paper](https://arxiv.org/abs/2008.06872) for more details.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/JY9t4xUAouk/0.jpg)](https://www.youtube.com/watch?v=JY9t4xUAouk)


_**Note**_: this repository is a re-implementation of the original framework using public components made by the same author after the end of internship.
It **doesn't contain** the original multi-subject training data and code and uses full mesh rasterizations as inputs rather than point projections.

It is based on a _**simple idea**_: select k fixed points in space and compute vectors from  these basis points to the nearest
points in a point cloud; use these vectors (or simply their norms) as features:

#![Teaser Image](bps.gif)


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

## Usage


### Installation


```
pip3 install git+https://github.com/sergeyprokudin/smplpix
```

### Demos

Check one of the provided examples:

- **ModelNet40 3D shape classification with BPS-MLP** (~89% accuracy, ~30 minutes of training on a non-GPU MacBook Pro, 
~3 minutes of training on Nvidia V100 16gb):

```
python bps_demos/train_modelnet_mlp.py
```



## License

This library is licensed under the MIT-0 License. See the LICENSE file.

