# Approach to training based on https://arxiv.org/pdf/1707.09775.pdf
import os
import ast

import numpy as np
import tensorflow as tf
import joblib

from .alexnet import AlexNet
from .utils.net.raschka_tf_utils import train as _train
from .utils.net.raschka_tf_utils import save


def train(config):
    """train neural network according to options in config

    Parameters
    ----------
    config : ConfigParser instance
        typically loaded by main function in __main__.py

    Returns
    -------
    None
    """
    # boilerplate to unpack config from DATA section
    set_sizes = config['DATA']['SET_SIZES']
    train_dir = config['DATA']['TRAIN_DIR']

    # get training data
    data_dict = joblib.load(config['DATA']['GZ_FILENAME'])
    x_train = data_dict['x_train']
    y_train = data_dict['y_train']
    training_set = [x_train, y_train]
    if config.has_option('DATA', 'VALIDATION_SIZE'):
        val_set = [data_dict['x_val'],
                   data_dict['y_val']]
    else:
        val_set = None

    # boilerplate to unpack config from TRAIN section
    net_name = int(config['TRAIN']['NETNAME'])
    number_nets_to_train = int(config['TRAIN']['number_nets_to_train'])
    input_shape = ast.literal_eval(config['TRAIN']['INPUT_SHAPE'])
    new_learn_rate_layers = ast.literal_eval(config['TRAIN']['NEW_LEARN_RATE_LAYERS'])
    base_learning_rate = float(config['TRAIN']['BASE_LEARNING_RATE'])
    new_layer_learning_rate = float(config['TRAIN']['NEW_LAYER_LEARNING_RATE'])
    epochs = int(config['TRAIN']['EPOCHS'])
    random_seed = int(config['TRAIN']['RANDOM_SEED'])
    batch_size = int(config['TRAIN']['BATCH_SIZE'])

    for net_number in range(number_nets_to_train):
        tf.reset_default_graph()
        with tf.Session() as sess:
            graph = tf.Graph()
            x = tf.placeholder(tf.float32, (None,) + alexnet_input_shape, name='x')
            y = tf.placeholder(tf.int32, shape=[None], name='y')
            y_onehot = tf.one_hot(indices=y, depth=len(np.unique(y_train)),
                                  dtype=tf.float32, name='y_onehot')

            if net_name == 'AlexNet':
                dropout_rate_placeholder = tf.placeholder(tf.float32, name='dropout_rate')
                model = AlexNet(x, init_layer=new_learn_rate_layers, dropout_rate=dropout_rate_placeholder)
                dropout_rate = 0.5
            elif net_name == 'VGG16':
                dropout_rate = None

            predictions = {
                'probabilities': tf.nn.softmax(layers_list[-1],  # last fully-connected
                                               name='probabilities'),
                'labels': tf.cast(tf.argmax(layers_list[-1], axis=1),
                                  tf.int32,
                                  name='labels')
            }

            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=layers_list[-1],
                                                        labels=y_onehot),
                name='cross_entropy_loss')

            var_list1 = []  # all layers before fully-connected
            var_list2 = []  # fully-connected layers
            for train_var in tf.trainable_variables():
                if any([new_rate_name in train_var.name
                        for new_rate_name in new_learn_rate_layers]):
                    var_list2.append(train_var)
                else:
                    var_list1.append(train_var)

            opt1 = tf.train.GradientDescentOptimizer(base_learning_rate)
            opt2 = tf.train.GradientDescentOptimizer(new_layer_learning_rate)
            grads = tf.gradients(cross_entropy_loss, var_list1 + var_list2)
            grads1 = grads[:len(var_list1)]
            grads2 = grads[len(var_list1):]
            train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
            train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
            train_op = tf.group(train_op1, train_op2, name='train_op')

            correct_predictions = tf.equal(predictions['labels'],
                                           y, name='correct_preds')
            saver = tf.train.Saver()

            accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, tf.float32),
                name='accuracy')

            _train(sess,
                   training_set=training_set,
                   validation_set=val_set,
                   initialize=True,
                   epochs=epochs,
                   shuffle=True,
                   random_seed=random_seed,
                   batch_size=batch_size,
                   droput_rate=dropout_rate)

            savepath = os.path.join(config['TRAIN']['MODEL_SAVE_PATH'],
                                    'net_number_{}'.format(net_number))
            save(saver, sess, epoch=epochs, path=savepath)
