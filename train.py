import os
import unet
import imgaug.augmenters as iaa
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
from utils import assign_paths

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

# CONNECT EVERYTHING
net = unet.UNet(IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS)
net.get_unet()
net.set_seq(seq)
net.process_train(train_im, train_seg)
net.process_val(val_im, val_seg)

# LOGGING
log_dir = 'logs/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq='batch')
callbacks = []  # [tensorboard_callback]

# FIT
# net.fit(batch=5, prefetch=5, repeat=5, epochs=2, callbacks=callbacks)
net.fit(net.trainset.shuffle(5).batch(5).repeat(), epochs=1,
        validation_data=net.valset.batch(5))

# SAVE THE MODEL
os.mkdir('saved_model')
net.model.save('saved_model/unet_trained')
