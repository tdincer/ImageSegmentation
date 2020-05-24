# UNet Tensorflow v2.2 Implementation

The code presented here is a Tensorflow v2.2 implementation of the original UNet model proposed in [Ronneberger et al. (2015)](Ronneberger2015.pdf).

In this implementation, the UNet model takes input from a Tensorflow input pipeline that allows image augmentation via the awesome [imgaug](https://github.com/aleju/imgaug) library.

This code can be trained on arbitrary imaging data. Check out the Jupyter notebook for the usage.

## Dependencies
- Tensorflow v2.0
- imgaug
- numpy
- scipy
- skimage
- matplotlib