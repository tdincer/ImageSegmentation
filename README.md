# UNet Tensorflow v2.2 Implementation

The code presented here is a Tensorflow v2.2 implementation of the original UNet model proposed in [Ronneberger et al. (2015)](Ronneberger2015.pdf).

In this implementation, the UNet model takes input from a Tensorflow input pipeline that allows image augmentation via the awesome [imgaug](https://github.com/aleju/imgaug) library.

## Dependencies
- Tensorflow v2.2
- imgaug
- numpy
- scipy
- skimage
- matplotlib

## Training
This code can be trained on arbitrary imaging data. The main parameters of the model, feeding the images, and augmentation pipeline can be easily done in the train.py file.

## Inference
After training, the model is saved as a h5 file and it can be used for inference within the inference.py file.

## Results
Here is an example prediction of the model on a test data:

![Unseen image during the training](./Result.jpg)

The prediction is not perfect but acceptable. It can be improved by better training the model.