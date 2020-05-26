import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.layers import (Input, Conv2D, Activation, BatchNormalization, Dropout, MaxPooling2D,
                                     UpSampling2D, Cropping2D, concatenate)
from tensorflow.keras.models import load_model
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class UNet:
    def __init__(self, imwidth=128, imheight=128, output_channels=2, autotune=-1):
        keras_backend.image_data_format = 'channels_last'
        self.model = None
        self.output_channels = output_channels
        self.imwidth = imwidth
        self.imheight = imheight
        self.trainset = None
        self.valset = None
        self.testset = None
        self.autotune = autotune
        self.seq = None
        self.experiment_name = None

    def init(self, experiment_name='awesome_project'):
        self.experiment_name = experiment_name

    @staticmethod
    def conv2d_block(input_tensor, filters, kernel_size=(3, 3), padding='valid', batch_norm=True):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(input_tensor)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # second layer
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   kernel_initializer='he_normal', padding=padding)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def get_unet(self, batch_norm=True, dropout_rate=0.1, optimizer='adam', metrics=None):
        if metrics is None:
            metrics = ['AUC']

        # Input = (N, 512, 512, 1)
        x = inputs = Input(shape=(self.imheight, self.imwidth, 1), dtype='float32')
        base = self.imheight

        # Contracting Path
        d1 = self.conv2d_block(input_tensor=x, filters=base / 8, kernel_size=(3, 3), padding='same',
                               batch_norm=batch_norm)
        p1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(d1)
        p1 = Dropout(rate=dropout_rate)(p1)

        d2 = self.conv2d_block(input_tensor=p1, filters=base / 4, kernel_size=(3, 3), padding='same',
                               batch_norm=batch_norm)
        p2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(d2)
        p2 = Dropout(rate=dropout_rate)(p2)

        d3 = self.conv2d_block(input_tensor=p2, filters=base / 2, kernel_size=(3, 3), padding='same',
                               batch_norm=batch_norm)
        p3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(d3)
        p3 = Dropout(rate=dropout_rate)(p3)

        d4 = self.conv2d_block(input_tensor=p3, filters=base, kernel_size=(3, 3), padding='same',
                               batch_norm=batch_norm)
        p4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(d4)
        p4 = Dropout(rate=dropout_rate)(p4)

        # Bottleneck
        bn = self.conv2d_block(input_tensor=p4, filters=base * 2, kernel_size=3, padding='same',
                               batch_norm=batch_norm)

        # Expansion Path
        u1 = UpSampling2D(size=(2, 2), interpolation='nearest')(bn)
        u1 = Conv2D(base, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(u1)
        cs = (d4.get_shape()[1] - u1.get_shape()[1]) // 2
        u1 = concatenate([Cropping2D(cropping=(cs, cs))(d4), u1])
        u1 = Dropout(rate=dropout_rate)(u1)
        u1 = self.conv2d_block(input_tensor=u1, filters=base, kernel_size=(3, 3), padding='same',
                               batch_norm=batch_norm)

        u2 = UpSampling2D(size=(2, 2), interpolation='nearest')(u1)
        u2 = Conv2D(base / 2, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(u2)
        cs = (d3.get_shape()[1] - u2.get_shape()[1]) // 2
        u2 = concatenate([Cropping2D(cropping=(cs, cs))(d3), u2])
        u2 = Dropout(rate=dropout_rate)(u2)
        u2 = self.conv2d_block(input_tensor=u2, filters=base / 2, kernel_size=(3, 3), padding='same',
                               batch_norm=batch_norm)

        u3 = UpSampling2D(size=(2, 2), interpolation='nearest')(u2)
        u3 = Conv2D(base / 4, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(u3)
        cs = (d2.get_shape()[1] - u3.get_shape()[1]) // 2
        u3 = concatenate([Cropping2D(cropping=(cs, cs))(d2), u3])
        u3 = Dropout(rate=dropout_rate)(u3)
        u3 = self.conv2d_block(input_tensor=u3, filters=base / 4, kernel_size=(3, 3), padding='same',
                               batch_norm=batch_norm)

        u4 = UpSampling2D(size=(2, 2), interpolation='nearest')(u3)
        u4 = Conv2D(base / 8, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(u4)
        cs = (d1.get_shape()[1] - u4.get_shape()[1]) // 2
        u4 = concatenate([Cropping2D(cropping=(cs, cs))(d1), u4])
        u4 = Dropout(rate=dropout_rate)(u4)
        u4 = self.conv2d_block(input_tensor=u4, filters=base / 8, kernel_size=(3, 3), padding='same',
                               batch_norm=batch_norm)

        if self.output_channels > 2:
            output_activation = 'softmax'
            loss = 'categorical_crossentropy'
        elif self.output_channels == 2:
            output_activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            raise ValueError

        outputs = Conv2D(self.output_channels, (1, 1), activation=output_activation, padding='valid',
                         kernel_initializer='he_normal')(u4)

        self.model = Model(inputs, outputs)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def decode_img(self, img):
        # convert the compressed string to a 1D uint8 tensor
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [self.imwidth, self.imheight], method='bilinear')
        img = tf.cast(tf.round(img), tf.uint8)
        return img

    def process_path_double(self, img_path, seg_path):
        # load the raw data from the file as a string
        img = tf.io.read_file(img_path)
        img = self.decode_img(img)
        seg = tf.io.read_file(seg_path)
        seg = self.decode_img(seg)
        # seg = tf.round(seg)
        return img, seg

    def process_path_single(self, img_path):
        img = tf.io.read_file(img_path)
        img = self.decode_img(img)
        return img

    def augment_batch(self, image, segmap):
        def augment_image(img, seg):
            img = img.numpy().reshape(img.numpy().shape[-3:])
            seg = seg.numpy().reshape(seg.numpy().shape[-3:])
            shape = img.shape
            segg = SegmentationMapsOnImage(seg, shape=shape)
            a, b = self.seq.augment(image=img, segmentation_maps=segg)
            return a, b.get_arr()

        img_shape = tf.shape(image)
        segmap_shape = tf.shape(segmap)
        image, segmap = tf.py_function(augment_image, [image, segmap], [tf.uint8, tf.uint8])
        image = tf.reshape(image, shape=img_shape)
        segmap = tf.reshape(segmap, shape=segmap_shape)
        return image, segmap

    def one_hot_label(self, img, seg):
        seg = tf.one_hot(tf.cast(seg, tf.uint8), self.output_channels)
        seg = tf.cast(seg, tf.float32)
        seg = tf.squeeze(seg, -2)
        return img, seg

    def normalize_double(self, img, seg):
        img = img / 255
        seg = seg / 255
        return img, seg

    def normalize_single(self, img):
        return img / 255

    def set_seq(self, seq):
        self.seq = seq

    def process_train(self, im, seg):
        self.trainset = tf.data.Dataset.from_tensor_slices((im, seg))
        self.trainset = self.trainset.map(self.process_path_double, num_parallel_calls=self.autotune)
        self.trainset = self.trainset.map(self.augment_batch, num_parallel_calls=self.autotune)
        self.trainset = self.trainset.map(self.normalize_double, num_parallel_calls=self.autotune)
        self.trainset = self.trainset.map(self.one_hot_label, num_parallel_calls=self.autotune)

    def process_val(self, im, seg):
        self.valset = tf.data.Dataset.from_tensor_slices((im, seg))
        self.valset = self.valset.map(self.process_path_double, num_parallel_calls=self.autotune)
        self.valset = self.valset.map(self.normalize_double, num_parallel_calls=self.autotune)
        self.valset = self.valset.map(self.one_hot_label, num_parallel_calls=self.autotune)

    def fit(self, trainset, validation_data, steps_per_epoch=100, epochs=1, callbacks=[]):
        self.model.fit(trainset, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks,
                       validation_data=validation_data)

    def load_model(self, file_path):
        self.model = load_model(file_path)
        self.imheight = self.model.input_shape[-3]
        self.imwidth = self.model.input_shape[-2]
        self.output_channels = self.model.output_shape[-1]
        self.autotune = -1

    def load_testset(self, im):
        if not isinstance(im, list):
            im = [im]
        self.testset = tf.data.Dataset.from_tensor_slices(im)
        self.testset = self.testset.map(self.process_path_single, num_parallel_calls=self.autotune)
        self.testset = self.testset.map(self.normalize_single, num_parallel_calls=self.autotune)

    # def fit(self, batch, prefetch, repeat, epochs, callbacks):
    #     self.model.fit(self.trainset.batch(batch).prefetch(prefetch).repeat(repeat),
    #                    validation_data=self.valset.batch(batch), epochs=epochs, callbacks=callbacks)
