import tensorflow as tf
from glob import glob
import numpy as np
from skimage.segmentation import find_boundaries


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


def decode_img(img, width, height):
    # convert the compressed string to a 1D uint8 tensor
    img = tf.image.decode_png(img, channels=1)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img = tf.image.resize(img, [width, height], method='bilinear')
    return img


def process_path(img_path, seg_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(img_path)
    img = decode_img(img)
    seg = tf.io.read_file(seg_path)
    seg = decode_img(seg)
    seg = tf.round(seg)
    # TODO: Remove the following line. To do this train_mask files should be corrected.
    seg = tf.image.rot90(seg, k=1, name=None)  # This is only for the test data.
    return img, seg


def assign_paths(base_dir, file_format='.png', split_no=5):
    train_im_files = glob(base_dir + '/*' + file_format)[:-split_no]
    train_seg_files = [x.replace('train/', 'train_mask/') for x in train_im_files]

    validation_im_files = glob(base_dir + '/*' + file_format)[-split_no:]
    validation_seg_files = [x.replace('train/', 'train_mask/') for x in validation_im_files]
    return train_im_files, train_seg_files, validation_im_files, validation_seg_files


def one_hot_label(img, seg, depth):
    seg = tf.one_hot(tf.cast(seg, tf.int32), depth)
    seg = tf.cast(seg, tf.float32)
    seg = tf.squeeze(seg, -2)
    return img, seg


def make_weight_map_batch(image, segmap):
    segmap_shape = tf.shape(segmap)
    weight_map = tf.transpose(segmap, [2, 0, 1])
    weight_map = tf.py_function(make_weight_map, [weight_map], [tf.float32])
    weight_map = tf.reshape(weight_map, shape=segmap_shape)
    return image, segmap, weight_map


def make_weight_map(masks):
    """
    Generate the weight maps as specified in the UNet paper
    for a set of binary masks.

    Parameters
    ----------
    masks: array-like
        A 3D array of shape (n_masks, image_height, image_width),
        where each slice of the matrix along the 0th axis represents one binary mask.

    Returns
    -------
    array-like
        A 2D array of shape (image_height, image_width)

    """
    w0 = 10
    sigma = 5
    masks = masks.numpy()
    nrows, ncols = masks.shape[1:]
    masks = (masks > 0) * 1
    distMap = np.zeros((nrows * ncols, masks.shape[0]))
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
    for i, mask in enumerate(masks):
        # find the boundary of each mask,
        # compute the distance of each pixel from this boundary
        bounds = find_boundaries(mask, mode='inner')
        X2, Y2 = np.nonzero(bounds)
        xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
        ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
        distMap[:, i] = np.sqrt(xSum + ySum).min(axis=0)
    ix = np.arange(distMap.shape[0])
    if distMap.shape[1] == 1:
        d1 = distMap.ravel()
        border_loss_map = w0 * np.exp((-1 * (d1 ** 2)) / (2 * (sigma ** 2)))
    else:
        if distMap.shape[1] == 2:
            d1_ix, d2_ix = np.argpartition(distMap, 1, axis=1)[:, :2].T
        else:
            d1_ix, d2_ix = np.argpartition(distMap, 2, axis=1)[:, :2].T
        d1 = distMap[ix, d1_ix]
        d2 = distMap[ix, d2_ix]
        border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
    xBLoss = np.zeros((nrows, ncols))
    xBLoss[X1, Y1] = border_loss_map
    # class weight map
    loss = np.zeros((nrows, ncols))
    w_1 = 1 - masks.sum() / loss.size
    w_0 = 1 - w_1
    loss[masks.sum(0) == 1] = w_1
    loss[masks.sum(0) == 0] = w_0
    ZZ = xBLoss + loss
    return ZZ


def load_data(img_folder, seg_folder):
    pass
