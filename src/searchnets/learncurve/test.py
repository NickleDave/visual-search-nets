import csv
import os

import numpy as np
import tensorflow as tf
import joblib
from tqdm import tqdm

from ..nets import AlexNet
from ..nets import VGG16

CSV_FIELDNAMES = ['setname', 'train_size', 'epochs', 'net_number', 'set_size', 'err']


def test(x_train,
         y_train,
         x_test,
         y_test,
         set_size_vec_train,
         set_size_vec_test,
         set_sizes,
         net_name,
         number_nets_to_train,
         input_shape,
         new_learn_rate_layers,
         epochs_list,
         train_size_list,
         batch_size,
         model_save_path,
         test_results_save_path):
    """test networks trained for a learning curve

    Parameters
    ----------
    x_train : numpy.ndarray
        training data, visual search stimuli
    y_train. numpy.ndarray
        expected outputs, vector of 0s and 1s (for 'target absent' and 'target present')
    x_test : numpy.ndarray
        test, visual search stimuli
    y_test. numpy.ndarray
        expected outputs for test set
    set_size_vec_train : numpy.ndarray
        vector whose length is equal to number of samples in x_train, each value indicates (visual search) set size for
        sample at corresponding index in x_train
    set_size_vec_test : numpy.ndarray
        vector whose length is equal to number of samples in x_test, each value indicates (visual search) set size for
        sample at corresponding index in x_test
    set_sizes : list
        of int, unique set of set sizes in data
    net_name : str
        name of convolutional neural net architecture to train.
        One of {'alexnet', 'VGG16'}
    number_nets_to_train : int
        number of training "replicates"
    input_shape : tuple
        with 3 elements, (rows, columns, channels)
        should be (227, 227, 3) for AlexNet
        and (224, 224, 3) for VGG16
    new_learn_rate_layers : list
        of layer names whose weights will be initialized randomly
        and then trained with the 'new_layer_learning_rate'.
    epochs_list : list
        of training epochs. Replicates will be trained for each
        value in this list. Can also just be one value, but a list
        is useful if you want to test whether effects depend on
        number of training epochs.
    train_size_list : list
        of number of samples in training set. Used to generate a learning curve
        (where x axis is size of training set and y axis is accuracy)
    batch_size : int
        number of samples in a batch of training data
    model_save_path : str
        path to directory where model checkpoints should be saved
    test_results_save_path : str
        path to directory where results from measuring accuracy on test set should be saved

    Returns
    -------
    None
    """
    csv_rows = []
    results_dict = {}

    for setname, x_data, y_data, set_size_vec in zip(
            ['train', 'test'],
            [x_train, x_test],
            [y_train, y_test],
            [set_size_vec_train, set_size_vec_test]):

        for train_size in train_size_list:
            for epochs in epochs_list:
                print(f'measuring accuracy on {setname} set for {net_name} model trained for {epochs} epochs'
                      f'with training set of {train_size} samples')

                err_per_set_size_per_model = []
                err_per_set_size_model_dict = {}
                predictions_per_model_dict = {}

                for net_number in range(number_nets_to_train):
                    tf.reset_default_graph()
                    graph = tf.Graph()
                    with tf.Session(graph=graph) as sess:
                        x = tf.placeholder(tf.float32, (None,) + input_shape, name='x')
                        rate = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), shape=(),
                                                           name='dropout_rate')

                        if net_name == 'alexnet':
                            model = AlexNet(x, init_layer=new_learn_rate_layers, dropout_rate=rate)
                        elif net_name == 'VGG16':
                            model = VGG16(x, init_layer=new_learn_rate_layers, dropout_rate=rate)

                        predictions = {
                            'probabilities': tf.nn.softmax(model.output,  # last fully-connected
                                                           name='probabilities'),
                            'labels': tf.cast(tf.argmax(model.output, axis=1),
                                              tf.int32,
                                              name='labels')
                        }

                        saver = tf.train.Saver()
                        restore_path = os.path.join(model_save_path,
                                                    f'training_set_with_{train_size}_samples',
                                                    f'trained_{epochs}_epochs',
                                                    f'net_number_{net_number}')
                        print(f'Loading model from {restore_path}')
                        ckpt_path = os.path.join(restore_path, f'{net_name}-model.ckpt-{epochs}')
                        saver.restore(sess, ckpt_path)

                        y_pred_all = []
                        batch_inds = np.arange(0, x_data.shape[0], batch_size)
                        total = len(batch_inds)
                        pbar = tqdm(enumerate(batch_inds), total=total)
                        for count, ind in pbar:
                            pbar.set_description(f'predicting target present/absent for batch {count} of {total}')
                            feed = {x: x_data[ind:ind + batch_size, :, :]}
                            y_pred = sess.run(predictions['labels'], feed_dict=feed)
                            y_pred_all.append(y_pred)
                        y_pred_all = np.concatenate(y_pred_all)
                        acc = np.asscalar(
                            np.sum(y_pred_all == y_data) / y_data.shape
                        )
                        err = 1.0 - acc
                        row = [setname, train_size,  epochs, net_number, 'all', err]
                        csv_rows.append(row)

                        err_per_set_size = []
                        for set_size in set_sizes:
                            # in line below, [0] at end because np.where returns a tuple
                            inds = np.where(set_size_vec == set_size)[0]
                            acc_this_set_size = np.asscalar(
                                np.sum(y_pred_all[inds] == y_data[inds]) / y_data[inds].shape
                            )
                            err_this_set_size = 1.0 - acc_this_set_size
                            err_per_set_size.append(err_this_set_size)
                            row = [setname, train_size,  epochs, net_number, set_size, err_this_set_size]
                            csv_rows.append(row)

                        # insert into dictionary where model name is key
                        # and list of accuracies per set size is the "value"
                        err_per_set_size_model_dict[restore_path] = err_per_set_size
                        predictions_per_model_dict[restore_path] = y_pred_all
                        # insert into dictionary where model name is key
                        # and list of accuracies per set size is the "value"
                        err_per_set_size_model_dict[restore_path] = err_per_set_size
                        predictions_per_model_dict[restore_path] = y_pred_all

                        # insert into dictionary where model name is key
                        # and list of accuracies per set size is the "value"
                        err_per_set_size_model_dict[restore_path] = err_per_set_size
                        predictions_per_model_dict[restore_path] = y_pred_all

        err_per_set_size_per_model = np.asarray(err_per_set_size_per_model)
        err_per_set_size_per_model = np.squeeze(err_per_set_size_per_model)

        results_dict_this_set = dict(err_per_set_size_per_model=err_per_set_size_per_model,
                                     err_per_set_size_model_dict=err_per_set_size_model_dict,
                                     predictions_per_model_dict=predictions_per_model_dict)
        results_dict[setname] = results_dict_this_set

    if not os.path.isdir(test_results_save_path):
        os.makedirs(test_results_save_path)
    results_fname = os.path.join(test_results_save_path,
                                 f'learncurve_{net_name}.gz')
    joblib.dump(results_dict, results_fname)

    results_csv = os.path.join(test_results_save_path,
                               f'learncurve_{net_name}.csv')

    with open(results_csv, 'w') as f:
        writer = csv.DictWriter(f=f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in csv_rows:
            row_dict = dict(zip(CSV_FIELDNAMES, row))
            writer.writerow(row_dict)
