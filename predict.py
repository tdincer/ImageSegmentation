import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, colorbar, assign_paths
from tensorflow.keras.models import load_model
import unet
import tensorflow as tf

model_file = 'unet.h5'

# DEFINE THE BASE DIRECTORY
base_dir = '/Users/tdincer/ML/NN_exercises/UNET'
train_im_folder = base_dir + '/data/train'

# FEED THE PATHS
_, _, val_im, val_seg = assign_paths(train_im_folder, file_format='.png', split_no=5)


net = unet.UNet()
net.model = load_model(model_file)

net.process_val(val_im, val_seg)
img, seg = next(iter(net.valset.shuffle(1)))
res = net.model.predict(tf.reshape(img, [1,128,128,1]))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
x1 = ax1.imshow(img.numpy().reshape(128, 128))
colorbar(x1)
x2 = ax2.imshow(np.argmax(seg, -1).reshape(128, 128))
colorbar(x2)
x3 = ax3.imshow(np.argmax(res, -1).reshape(128,128))
colorbar(x3)
ax1.set_title('Input')
ax2.set_title('Ground Truth')
ax3.set_title('Prediction')
plt.show()
