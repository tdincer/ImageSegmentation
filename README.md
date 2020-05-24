# Image Segmentation Model

Image segmentation model is an implementation of UNet ([Ronneberger2015](http://lmb.informatik.uni-freiburg.de/)) using
Tensorflow v2.2. The network can be trained to perform image segmentation on arbitrary imaging data. Check out the
Jupyter notebook for the usage.



## Features

Data is pipelined to the model with Tensorflow Data (tf.data) API.

Image augmentatiton is available with [imgaug](https://github.com/aleju/imgaug).

## Model
The model is a fully convolutional network; therefore the shape of the input images is not predefined. The output layer is activated with a sigmoid in the case of a binary label and with a softmax in the case of multiple label. In the former, the default loss function is a binary cross entropy.

## Results

![Result](UNet_validation.png)

## Dependencies
- Tensorflow v2.0
- imgaug
- numpy
- scipy
- skimage
- matplotlib


### TODO:
  At some point between image resizing or augmentatation, the input image is getting out of 0-1 range.