"""utility functions for tf.data.Datasets"""
import numpy as np
import tensorflow as tf

IMAGENET_MEANS = np.asarray([123.68, 116.779, 103.939])  # in RGB order


def load(x):
    x = tf.read_file(x)
    x = tf.image.decode_png(x)
    x = tf.cast(x, tf.float32)
    return x


def preprocess(x, convert_bgr=True):
    x = x - IMAGENET_MEANS  # do this before flipping channel order
    if convert_bgr:
        # make RGB into BGR
        x = tf.reverse(x, axis=[-1])
    return x


def _generic_dataset(x, y, preprocess_func, batch_size, shuffle=True):
    if type(x) == list:
        x = np.asarray(x)
    x_ds = tf.data.Dataset.from_tensor_slices(x)
    x_ds = x_ds.map(load)
    x_ds = x_ds.map(preprocess_func)
    y_ds = tf.data.Dataset.from_tensor_slices(y)
    ds = tf.data.Dataset.zip((x_ds, y_ds))
    if shuffle:
        ds = ds.shuffle(buffer_size=x.shape[-1])
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=x.shape[-1])
    return ds


def alexnet_preprocess():
    def anonymous(x):
        return preprocess(x, convert_bgr=True)
    return anonymous


def alexnet_dataset(x, y, batch_size, shuffle=True):
    return _generic_dataset(x, y, alexnet_preprocess(), batch_size, shuffle)


def vgg16_preprocess():
    def anonymous(x):
        return preprocess(x, convert_bgr=False)  # because weights we're using have channels in 1st layer switched
    return anonymous


def vgg16_dataset(x, y, batch_size, shuffle=True):
    return _generic_dataset(x, y, vgg16_preprocess(), batch_size, shuffle)


def get_dataset(x, y, net_name, batch_size, shuffle=True):
    if net_name == 'alexnet':
        return alexnet_dataset(x, y, batch_size, shuffle)
    elif net_name == 'VGG16':
        return vgg16_dataset(x, y, batch_size, shuffle)
