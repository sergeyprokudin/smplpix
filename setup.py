"""Setup for the project."""
from setuptools import setup

setup(
    name="smplpix",
    version=1.0,
    description="SMPLpix: Neural Avatars from Deformable 3D models",
    install_requires=["torch", "torchvision", "trimesh", "pyrender"],
    author="Sergey Prokudin",
    license="MIT",
    author_email="sergey.prokudin@gmail.com",
    packages=["smplpix"]
)