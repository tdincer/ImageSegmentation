import unet
import utils
import tensorflow as tf
import matplotlib.pyplot as plt
from augmenter import augment_batch

# INPUT CHARACTERISTICS
OUTPUT_CHANNELS = 2
IMG_WIDTH = 128
IMG_HEIGHT = 128

# LOAD THE MODEL
AUTOTUNE = 1
BATCH_SIZE = 4
net = unet.UNet(IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS)
net.get_unet()

# DEFINE THE BASE DIRECTORY
base_dir = '/Users/tdincer/ML/NN_exercises/UNET'
train_im_folder = base_dir + '/data/train'

# MAKE THE PATHS
train_im, train_seg, validation_im, validation_seg = utils.assign_paths(train_im_folder, file_format='.png', split_no=5)


# DEFINE THE TRAIN DATA FLOW
tf_trainset = tf.data.Dataset.from_tensor_slices((train_im, train_seg))
tf_trainset = tf_trainset.map(utils.process_path, num_parallel_calls=AUTOTUNE)
tf_trainset = tf_trainset.map(augment_batch, num_parallel_calls=AUTOTUNE)
tf_trainset = tf_trainset.map(utils.one_hot_label, num_parallel_calls=AUTOTUNE)

# DEFINE THE AUGMENTATION DATA FLOW
tf_valset = tf.data.Dataset.from_tensor_slices((validation_im, validation_seg))
tf_valset = tf_valset.map(utils.process_path, num_parallel_calls=AUTOTUNE)
tf_valset = tf_valset.map(utils.one_hot_label, num_parallel_calls=AUTOTUNE)


net.model.fit(tf_trainset.batch(5).prefetch(5).repeat(1), validation_data=tf_valset.batch(5), epochs=5)






epochs = 10
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))
    # Iterate over the batches of the dataset.
    for step, (img, seg, cw) in enumerate(tf_augmented.batch(10).prefetch(10).repeat(10)):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        net.model.fit(img, seg, sample_weight=cw)



# tf_augmented = tf_augmented.batch(5).repeat(10)

# img, seg, cw = next(iter(tf_augmented))

# img, seg = next(iter(tf_dataset))
# img_aug, seg_aug = next(iter(tf_augmented))
# wmap_aug = next(iter(weight_map))

# img, seg = next(iter(tf_dataset))
# img_aug, seg_aug = augment_batch(img, seg)
# wmap_aug = make_weight_map_batch(img_aug, seg_aug)


def plot():
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    ax1.imshow(img.numpy().reshape(IMG_WIDTH, IMG_HEIGHT))
    ax2.imshow(seg.numpy().reshape(IMG_WIDTH, IMG_HEIGHT))
    ax3.imshow(img_aug.numpy().reshape(IMG_WIDTH, IMG_HEIGHT))
    ax4.imshow(seg_aug.numpy().reshape(IMG_WIDTH, IMG_HEIGHT))
    ax5.imshow(img_aug.numpy().reshape(IMG_WIDTH, IMG_HEIGHT))
    ax6.imshow(wmap_aug.numpy().reshape(IMG_WIDTH, IMG_HEIGHT))

    plt.show()

plot()


img, seg = next(iter(tf_validation))
res = net.model.predict(img.numpy().reshape(1,128,128,1))

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

x1 = ax1.imshow(img.numpy().reshape(128, 128))
colorbar(x1)

x2 = ax2.imshow(np.argmax(seg, -1).reshape(128, 128))
colorbar(x2)

x3 = ax3.imshow(np.argmax(res, -1).reshape(128,128))
colorbar(x3)
ax1.set_title('Image')
ax2.set_title('Label')
ax3.set_title('Prediction')
plt.show()