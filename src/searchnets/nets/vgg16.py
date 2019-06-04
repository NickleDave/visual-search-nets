"""VGG16 implementation, adapted from Frederik Kratzert AlexNet implementation, under BSD-3 license
https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/master/LICENSE

Uses VGG16 architecture as specified in
https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/keras/_impl/keras/applications/vgg16.py
which is released under Apache license 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""
import os

import tensorflow as tf
import numpy as np

from ..utils.figshare import fetch
from ..utils.figshare_urls import VGG16_WEIGHTS_URL
from .layers import max_pool, dropout

THIS_FILE_PATH = os.path.dirname(__file__)


class VGG16:
    """Implementation of VGG16."""

    def __init__(self, x, init_layer, dropout_rate, num_classes=2, weights_path='DEFAULT'):
        """initialize an instance of VGG16

        Parameters
        ----------
        x: tensorflow.Placeholder
            Placeholder for the input tensor.
        init_layer: list
            of strings, names of layers that will have variables initialized at random and
            trained "from scratch" instead of loading pre-trained weights.
        dropout_rate: tensorflow.Placeholder
            Dropout probability. Default used by 'train' function is 0.5.
        num_classes: int
            Number of classes in the dataset. Default is 2 (for "target present" / "target absent")
        weights_path: str
            Complete path to the pre-trained weight file. If file doesn't exist, weights will
            be downloaded and saved to this location. Default is 'DEFAULT', in which case the
            path used is '../../../data/neural_net_weights/bvlc_alexnet.npy'
        """
        self.x = x
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.init_layer = init_layer

        if weights_path == 'DEFAULT':
            weights_path = os.path.join(
                THIS_FILE_PATH,
                '..', '..', '..',
                'data', 'neural_net_weights', 'vgg16_weights.npz'
            )
        if not os.path.isfile(weights_path):
            print("downloading weights for AlexNet")
            fetch(url=VGG16_WEIGHTS_URL,
                  destination_path=weights_path)
        self.weights_path = weights_path
        self.weights_dict = np.load(self.weights_path)

        self.output = None  # will be set during call to self.create() below

        self.create()

    def conv(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
             padding='SAME', groups=1):
        """Create a convolution layer.

        Adapted from: https://github.com/ethereon/caffe-tensorflow
        """
        # Get number of input channels
        input_channels = int(x.get_shape()[-1])

        # Create lambda function for the convolution
        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                             strides=[1, stride_y, stride_x, 1],
                                             padding=padding)

        with tf.variable_scope(name) as scope:
            # if this is a layer we should initialize new weights for
            if name in self.init_layer:
                weights = tf.get_variable('weights', shape=[filter_height,
                                                            filter_width,
                                                            input_channels / groups,
                                                            num_filters])
                biases = tf.get_variable('biases', shape=[num_filters])
            else:
                # if this is a layer we should load weights for
                weights = tf.Variable(self.weights_dict[name + '_W'], name='weights')
                biases = tf.Variable(self.weights_dict[name + '_b'], name='biases')

        if groups == 1:
            conv = convolve(x, weights)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                     value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu

    def fc(self, x, num_in, num_out, name, relu=True):
        """Create a fully connected layer."""
        with tf.variable_scope(name) as scope:
            if name in self.init_layer:
                weights = tf.get_variable('weights', shape=[num_in, num_out],
                                          trainable=True)
                biases = tf.get_variable('biases', [num_out], trainable=True)
            else:
                weights = tf.Variable(self.weights_dict[name + '_W'], name='weights')
                biases = tf.Variable(self.weights_dict[name + '_b'], name='biases')

            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

    def create(self):
        """Create the network graph."""
        block1_conv1 = self.conv(self.x, 3, 3, 64, 1, 1, padding='SAME', name='conv1_1')
        block1_conv2 = self.conv(block1_conv1, 3, 3, 64, 1, 1, padding='SAME', name='conv1_2')
        block1_pool = max_pool(block1_conv2, 2, 2, 2, 2, padding='VALID', name='conv1__pool')

        block2_conv1 = self.conv(block1_pool, 3, 3, 128, 1, 1, padding='SAME', name='conv2_1')
        block2_conv2 = self.conv(block2_conv1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_2')
        block2_pool = max_pool(block2_conv2, 2, 2, 2, 2, padding='VALID', name='conv2_pool')

        block3_conv1 = self.conv(block2_pool, 3, 3, 256, 1, 1, padding='SAME', name='conv3_1')
        block3_conv2 = self.conv(block3_conv1, 3, 3, 256, 1, 1, padding='SAME', name='conv3_2')
        block3_conv3 = self.conv(block3_conv2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_3')
        block3_pool = max_pool(block3_conv3, 2, 2, 2, 2, padding='VALID', name='block3_pool')

        block4_conv1 = self.conv(block3_pool, 3, 3, 512, 1, 1, padding='SAME', name='conv4_1')
        block4_conv2 = self.conv(block4_conv1, 3, 3, 512, 1, 1, padding='SAME', name='conv4_2')
        block4_conv3 = self.conv(block4_conv2, 3, 3, 512, 1, 1, padding='SAME', name='conv4_3')
        block4_pool = max_pool(block4_conv3, 2, 2, 2, 2, padding='VALID', name='conv4_pool')

        block5_conv1 = self.conv(block4_pool, 3, 3, 512, 1, 1, padding='SAME', name='conv5_1')
        block5_conv2 = self.conv(block5_conv1, 3, 3, 512, 1, 1, padding='SAME', name='conv5_2')
        block5_conv3 = self.conv(block5_conv2, 3, 3, 512, 1, 1, padding='SAME', name='conv5_3')
        block5_pool = max_pool(block5_conv3, 2, 2, 2, 2, padding='VALID', name='conv5_pool')

        new_shape = int(np.prod(block5_pool.get_shape()[1:]))
        flattened = tf.reshape(block5_pool, [-1, new_shape], name='flatten')
        fc6 = self.fc(flattened, new_shape, 4096, name='fc6')
        dropout1 = dropout(fc6, self.dropout_rate)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = self.fc(dropout1, 4096, 4096, name='fc7')
        dropout2 = dropout(fc7, self.dropout_rate)

        # 8th Layer: FC and return unscaled activations
        fc8 = self.fc(dropout2, 4096, self.num_classes, relu=False, name='fc8')

        self.output = fc8
