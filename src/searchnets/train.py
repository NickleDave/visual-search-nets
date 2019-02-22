# Approach to training based on https://arxiv.org/pdf/1707.09775.pdf
import os
import ast

import numpy as np
import tensorflow as tf
import joblib

from .alexnet import AlexNet
from .vgg16 import VGG16
from .utils.net.raschka_tf_utils import batch_generator
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
    net_name = config['TRAIN']['NETNAME']
    number_nets_to_train = int(config['TRAIN']['number_nets_to_train'])
    input_shape = ast.literal_eval(config['TRAIN']['INPUT_SHAPE'])
    new_learn_rate_layers = ast.literal_eval(config['TRAIN']['NEW_LEARN_RATE_LAYERS'])
    base_learning_rate = float(config['TRAIN']['BASE_LEARNING_RATE'])
    new_layer_learning_rate = float(config['TRAIN']['NEW_LAYER_LEARNING_RATE'])
    epochs = int(config['TRAIN']['EPOCHS'])
    batch_size = int(config['TRAIN']['BATCH_SIZE'])
    random_seed = int(config['TRAIN']['RANDOM_SEED'])

    np.random.seed(random_seed)  # for shuffling in batch_generator
    tf.random.set_random_seed(random_seed)

    for net_number in range(number_nets_to_train):
        tf.reset_default_graph()
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            x = tf.placeholder(tf.float32, (None,) + input_shape, name='x')
            y = tf.placeholder(tf.int32, shape=[None], name='y')
            y_onehot = tf.one_hot(indices=y, depth=len(np.unique(y_train)),
                                  dtype=tf.float32, name='y_onehot')

            if net_name == 'alexnet':
                rate = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), shape=(), name='dropout_rate')
                model = AlexNet(x, init_layer=new_learn_rate_layers, dropout_rate=rate)
                dropout_rate = 0.5
            elif net_name == 'VGG16':
                model = VGG16()
                dropout_rate = 1.0

            predictions = {
                'probabilities': tf.nn.softmax(model.output, name='probabilities'),
                'labels': tf.cast(tf.argmax(model.output, axis=1), tf.int32, name='labels')
            }

            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=model.output,
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

            X_data = np.array(training_set[0])
            y_data = np.array(training_set[1])
            training_loss = []

            sess.run(tf.global_variables_initializer())

            for epoch in range(1, epochs + 1):
                batch_gen = batch_generator(X_data, y_data,
                                            batch_size=batch_size,
                                            shuffle=True)
                avg_loss = 0.0
                for i, (batch_x, batch_y) in enumerate(batch_gen):
                    feed = {x: batch_x,
                            y: batch_y,
                            rate: dropout_rate}

                    loss, _ = sess.run(
                        [cross_entropy_loss, train_op],
                        feed_dict=feed)
                    avg_loss += loss

                training_loss.append(avg_loss / (i + 1))
                print('Epoch %02d Training Avg. Loss: %7.3f' % (
                    epoch, avg_loss), end=' ')
                if val_set is not None:
                    feed = {x: val_set[0],
                            y: val_set[1]}
                    valid_acc = sess.run(accuracy, feed_dict=feed)
                    print(' Validation Acc: %7.3f' % valid_acc)
                else:
                    print()

            savepath = os.path.join(config['TRAIN']['MODEL_SAVE_PATH'],
                                    'net_number_{}'.format(net_number))
            save(saver, sess, epoch=epochs, path=savepath)
