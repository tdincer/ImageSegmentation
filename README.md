# UNet Tensorflow v2.1 Implementation

The code presented here is a Tensorflow v2.1 implementation of the original UNet model proposed in [Ronneberger2015](http://lmb.informatik.uni-freiburg.de/).

In this code, the model takes input from a Tensorflow input pipeline, which allows processing image augmentation via the awesome [imgaug](https://github.com/aleju/imgaug) library.

This code can be trained on arbitrary imaging data. Check out the Jupyter notebook for the usage.


## Results

![Result](UNet_validation.png)

## Dependencies
- Tensorflow v2.0
- imgaug
- numpy
- scipy
- skimage
- matplotlib