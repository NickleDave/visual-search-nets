import csv
import os

import numpy as np
import tensorflow as tf
import joblib
from tqdm import tqdm

from searchnets.nets import AlexNet
from searchnets.nets import VGG16

CSV_FIELDNAMES = ['setname', 'train_size', 'epochs', 'net_number', 'set_size', 'err']


def test(x_train, y_train, x_test, y_test, set_size_vec_train, set_size_vec_test,
         net_name, number_nets_to_train, set_sizes, input_shape, new_learn_rate_layers, epochs_list, train_size_list,
         batch_size, model_save_path, test_results_save_path):
    """test networks trained for a learning curve"""
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
