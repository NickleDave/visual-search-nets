import ast
import os

import numpy as np
import tensorflow as tf
import joblib

from .alexnet.myalexnet_forward_newtf import alexnet
from .utils.net.raschka_tf_utils import load, predict


def test(config):
    """test trained AlexNet models on test searchstims dataset

    Parameters
    ----------
    config

    Returns
    -------
    None

    saves .npz output file with following keys:
        acc_per_set_size_per_model : np.ndarray
            where rows are models and columns are set size
        acc_per_set_size_model_dict : dict
            where keys are paths to model and values are accuracy by set size.
            The actual set sizes are in the .npz file saved by data, under
            the key 'set_size_vec_test'.
        predictions_per_model_dict : dict
            where keys are paths to model and values are array
            of predictions made by that model for test set
    """
    # boilerplate to unpack config from DATA section
    set_sizes = config['DATA']['SET_SIZES']
    train_dir = config['DATA']['TRAIN_DIR']

    # get test data
    data_dict = joblib.load(config['DATA']['GZ_FILENAME'])
    x_test = data_dict['x_test']
    y_test = data_dict['y_test']

    # boilerplate to unpack config from TRAIN section
    number_nets_to_train = int(config['TRAIN']['NUMBER_NETS_TO_TRAIN'])
    alexnet_input_shape = ast.literal_eval(config['TRAIN']['ALEXNET_INPUT_SHAPE'])
    epochs = int(config['TRAIN']['EPOCHS'])
    batch_size = int(config['TRAIN']['BATCH_SIZE'])

    acc_per_set_size_per_model = []
    acc_per_set_size_model_dict = {}
    predictions_per_model_dict = {}
    for net_number in range(number_nets_to_train):
        tf.reset_default_graph()
        with tf.Session() as sess:
            graph = tf.Graph()
            x = tf.placeholder(tf.float32, (None,) + alexnet_input_shape, name='x')
            y = tf.placeholder(tf.int32, shape=[None], name='y')
            y_onehot = tf.one_hot(indices=y, depth=len(np.unique(y_test)),
                                  dtype=tf.float32, name='y_onehot')

            layers_list = alexnet(graph, x)

            predictions = {
                'probabilities': tf.nn.softmax(layers_list[-1],  # last fully-connected
                                               name='probabilities'),
                'labels': tf.cast(tf.argmax(layers_list[-1], axis=1),
                                  tf.int32,
                                  name='labels')
            }

            saver = tf.train.Saver()

            savepath = os.path.join(config['TRAIN']['MODEL_SAVE_PATH'],
                                    'net_number_{}'.format(net_number))

            load(saver=saver, sess=sess, path=savepath, epoch=epochs)

            predictions = []
            batch_inds = np.arange(0, x_test.shape[0], batch_size)
            for count, ind in enumerate(batch_inds):
                print('predicting target present/absent for batch {} of {}'
                      .format(count, len(batch_inds)))
                predictions.append(predict(sess,
                                           x_test[ind:ind + batch_size, :, :],
                                           return_proba=False))
            predictions = np.concatenate(predictions)
            set_sizes = data_dict['set_sizes']
            set_size_vec_test = data_dict['set_size_vec_test']
            acc_per_set_size = []
            for set_size in set_sizes:
                # in line below, [0] at end because np.where returns a tuple
                inds = np.where(set_size_vec_test == set_size)[0]
                acc_this_set_size = (np.sum(predictions[inds] == y_test[inds]) /
                                     y_test[inds].shape)
                acc_per_set_size.append(acc_this_set_size)

            # append to list of lists which we convert into matrix
            acc_per_set_size_per_model.append(acc_per_set_size)
            # and insert into dictionary where model name is key
            # and list of accuracies per set size is the "value"
            acc_per_set_size_model_dict[savepath] = acc_per_set_size
            predictions_per_model_dict[savepath] = predictions

    acc_per_set_size_per_model = np.asarray(acc_per_set_size_per_model)
    acc_per_set_size_per_model = np.squeeze(acc_per_set_size_per_model)

    savepath = config['TEST']['TEST_RESULTS_SAVE_PATH']
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    results_fname = os.path.join(savepath, 'test_alexnet_output.gz')
    results_dict = dict(acc_per_set_size_per_model=acc_per_set_size_per_model,
                        acc_per_set_size_model_dict=acc_per_set_size_model_dict,
                        predictions_per_model_dict=predictions_per_model_dict)
    joblib.dump(results_dict, results_fname)
