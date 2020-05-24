import os
import unet
import imgaug.augmenters as iaa
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
from utils import colorbar, assign_paths

# INPUT CHARACTERISTICS
OUTPUT_CHANNELS = 2
IMG_WIDTH = 128
IMG_HEIGHT = 128

# DEFINE THE BASE DIRECTORY
base_dir = '/Users/tdincer/ML/NN_exercises/UNET'
train_im_folder = base_dir + '/data/train'

# FEED THE PATHS
train_im, train_seg, val_im, val_seg = assign_paths(train_im_folder, file_format='.png', split_no=5)

# CONSTRUCT THE AUGMENTOR
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.ElasticTransformation(alpha=10, sigma=3),
])

# CONNECT EVERYTHING & FIT
net = unet.UNet(IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS)
net.get_unet()
net.set_seq(seq)
net.process_train(train_im, train_seg)
net.process_val(val_im, val_seg)

# LOGGING
log_dir = 'logs/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq='batch')
callbacks = [tensorboard_callback]

# FIT
net.fit(batch=5, prefetch=5, repeat=50, epochs=5, callbacks=callbacks)

# SAVE THE MODEL
os.mkdir('saved_model')
net.model.save('saved_model/unet_trained')


# INFERENCE AND FIGURES
img, seg = next(iter(net.valset.shuffle(5).take(1)))
res = net.model.predict(img.numpy().reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
x1 = ax1.imshow(img.numpy().reshape(IMG_WIDTH, IMG_HEIGHT))
colorbar(x1)
x2 = ax2.imshow(np.argmax(seg, -1).reshape(IMG_WIDTH, IMG_HEIGHT))
colorbar(x2)
x3 = ax3.imshow(np.argmax(res, -1).reshape(IMG_WIDTH, IMG_HEIGHT))
colorbar(x3)
ax1.set_title('Image')
ax2.set_title('Ground Truth')
ax3.set_title('Prediction')
plt.show()
