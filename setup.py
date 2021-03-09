"""Setup for the project."""
from setuptools import setup

setup(
    name="unet",
    version=1.0,
    description="Customized implementation of the U-Net in PyTorch",
    install_requires=["torch"],
    author="Alexandre Milesi",
    license="GNU",
    author_email="alexandre.milesi@etu.utc.fr",
    packages=["unet"]
)