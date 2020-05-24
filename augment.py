import imageio
import numpy as np
import imgaug as ia
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from imgaug.augmentables.batches import Batch
from scipy import ndimage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def plot_image_segmap(im=0):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(images[im], cmap='gray', vmin=0, vmax=255)
    ax2.imshow(segmaps[im], cmap='gray', vmin=0, vmax=255)

    ax1.set_title('Image')
    ax2.set_title('Segmentation Map')
    plt.show()


def plot_augmented(im=0, shape=(512, 512)):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(batches_aug[im].images_aug[im].reshape(shape))
    ax2.imshow(batches_aug[im].segmentation_maps_aug[im].get_arr().reshape(shape))
    plt.show()


def create_generator(lst):
    for list_entry in lst:
        yield list_entry


# Load the data: 30 images & 30 segmentation maps.
imfile = './data/train-volume.tif'
segfile = './data/train-labels.tif'
images = imageio.volread(imfile)
segmaps = imageio.volread(segfile)

# Rotate the segmentation maps to their correct position & adjust the shapes for imgaug.
segmaps = np.array([ndimage.rotate(x, 90) for x in segmaps])
images = [x.reshape(512, 512, 1) for x in images]
segmaps = [x.reshape(512, 512, 1) for x in segmaps]
segmaps = [SegmentationMapsOnImage(x, shape=images[0].shape) for x in segmaps]

# Combine the images and the segmentation maps to Batch instance.
NB_BATCHES = 30
batches = [Batch(images=images, segmentation_maps=segmaps) for _ in range(NB_BATCHES)]

# Make the sequential model
seq = iaa.Sequential([
    iaa.ElasticTransformation(alpha=90, sigma=9),  # water-like effect
    # iaa.Affine(rotate=(-175, 175)),
], random_order=True)


# Augmentation
# Here, I show 3 different ways of augmenting the image + segmentation map data.

# 1. Using multicore, meaning that background=True. This method takes the batch and creates the new files at once.
def type1():
    batches_aug = list(seq.augment_batches(batches, background=True))
    return batches_aug


# 2. Using pooling with more control.
# processes=-1: Use all CPU except 1.
# maxtasksperchild=20: Restart the child processes after 20 tasks.
def type2():
    with seq.pool(processes=-1, maxtasksperchild=20, seed=1) as pool:
        batches_aug = pool.map_batches(batches)
        ia.imshow(batches_aug[0].images_aug[0].reshape(512, 512))


# 3. Pool with generators
# Same as above but map_batches replaced by imap_batches. The output of imap_batches is a generator.
# output_buffer_size=5 restricts the allowed number of waiting batches to 5.
def type3():
    my_generator = create_generator(batches)

    with seq.pool(processes=-1, seed=1) as pool:
        batches_aug = pool.imap_batches(my_generator, output_buffer_size=5)

        for i, batch_aug in enumerate(batches_aug):
            ia.imshow(batch_aug.images_aug[1].reshape(512, 512))
            # if i == 0:
            #    ia.imshow(batch_aug.images_aug[0].reshape(512, 512))
            # do something else with the batch here


batches_aug = type1()
plot_augmented()
