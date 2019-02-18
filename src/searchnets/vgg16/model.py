# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""VGG16 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image
Recognition](https://arxiv.org/abs/1409.1556)
"""
import os

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG16(input_tensor=None,
          pooling=None,
          classes=2):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    Arguments:
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    # Determine proper input shape
    input_shape = (224, 224, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
        img_input)
    x = Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1')(
        x)
    x = Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1')(
        x)
    x = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2')(
        x)
    x = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1')(
        x)
    x = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2')(
        x)
    x = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1')(
        x)
    x = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2')(
        x)
    x = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    INIT_CLASSES = 1000
    x = Dense(INIT_CLASSES, activation='softmax', name='predictions')(x)

    # Create model.
    model = Model(img_input, x, name='vgg16')

    weights_path = get_file(
        'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
        WEIGHTS_PATH,
        cache_subdir='models',
        file_hash='64373286793e3c8b2b4e3219cbf3544b')

    model.load_weights(weights_path)

    for layer in range(4):
        # as per Poder 2017, we don't want trained weights for last 3 layers
        model.layers.pop()

    x = Dense(4096, activation='relu', name='fc1')(model.layers[-1].output)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model.
    model = Model(img_input, x, name='vgg16')

    # and we only want to train the final output layer
    for layer in model.layers[:-1]:
        layer.trainable = False

    return model
