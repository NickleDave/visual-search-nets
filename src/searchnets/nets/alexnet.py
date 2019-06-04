"""AlexNet implementation, adapted from Frederik Kratzert under BSD-3 license
https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/master/LICENSE
"""
import os

import tensorflow as tf
import numpy as np

from ..utils.figshare import fetch
from ..utils.figshare_urls import ALEXNET_WEIGHTS_URL
from .layers import max_pool, lrn, dropout

THIS_FILE_PATH = os.path.dirname(__file__)


class AlexNet:
    """Implementation of AlexNet."""

    def __init__(self, x, init_layer, dropout_rate, num_classes=2, weights_path='DEFAULT'):
        """Create the graph of the AlexNet model.

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
                'data', 'neural_net_weights', 'bvlc_alexnet.npy'
            )
        if not os.path.isfile(weights_path):
            print("downloading weights for AlexNet")
            fetch(url=ALEXNET_WEIGHTS_URL,
                  destination_path=weights_path)
        self.weights_path = weights_path
        with open(self.weights_path, "rb") as weights_fp:
            # use item to get dictionary saved in a numpy array
            self.weights_dict = np.load(weights_fp, encoding="latin1", allow_pickle=True).item()

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
                weights = tf.Variable(self.weights_dict[name][0], name='weights')
                biases = tf.Variable(self.weights_dict[name][1], name='biases')

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
                weights = tf.Variable(self.weights_dict[name][0], name='weights')
                biases = tf.Variable(self.weights_dict[name][1], name='biases')

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
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = self.conv(self.x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = self.conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = self.conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = self.conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = self.conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = self.fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.dropout_rate)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = self.fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.dropout_rate)

        # 8th Layer: FC and return unscaled activations
        fc8 = self.fc(dropout7, 4096, self.num_classes, relu=False, name='fc8')

        self.output = fc8
