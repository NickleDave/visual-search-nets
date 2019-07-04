import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import joblib
from tqdm import tqdm

from .nets import AlexNet, VGG16
from .utils.tfdata import get_dataset


def test(gz_filename,
         net_name,
         number_nets_to_train,
         input_shape,
         new_learn_rate_layers,
         epochs_list,
         batch_size,
         model_save_path,
         test_results_save_path,
         configfile):
    """measure accuracy of trained convolutional neural networks on test set of visual search stimuli

    Parameters
    ----------
    gz_filename : str
        name of .gz file containing prepared data sets
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
    batch_size : int
        number of samples in a batch of training data
    model_save_path : str
        path to directory where model checkpoints should be saved
    test_results_save_path : str
        path to directory where results from measuring accuracy on test set should be saved
    configfile : str
        filename of config.ini file. Used (without .ini extension) as name for output file
        that is saved in test_results_save_path.

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
    print('loading testing data')
    data_dict = joblib.load(gz_filename)

    set_sizes_by_stim_type = data_dict['set_sizes_by_stim_type']
    set_sizes = []
    for stim_type, set_sizes_this_stim in set_sizes_by_stim_type.items():
        if set_sizes == []:
            set_sizes = set_sizes_this_stim
        else:
            if set_sizes_this_stim != set_sizes:
                raise ValueError('set sizes are not the same across visual search stimuli')
            else:
                continue

    for epochs in epochs_list:
        print(f'measuring accuracy on test set for {net_name} model trained for {epochs} epochs')

        acc_per_set_size_per_model = []
        acc_per_set_size_model_dict = {}
        predictions_per_model_dict = {}

        for net_number in range(number_nets_to_train):
            tf.reset_default_graph()
            graph = tf.Graph()
            with tf.Session(graph=graph) as sess:
                filenames_placeholder = tf.placeholder(tf.string, shape=[None])
                labels_placeholder = tf.placeholder(tf.int64, shape=[None])
                test_ds = get_dataset(filenames_placeholder, labels_placeholder, net_name, batch_size,
                                      shuffle=False, shuffle_size=None)

                x = tf.placeholder(tf.float32, (None,) + input_shape, name='x')
                rate = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), shape=(), name='dropout_rate')

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
                                        f'trained_{epochs}_epochs',
                                        f'net_number_{net_number}')
                print(f'Loading model from {restore_path}')
                try:
                    ckpt_path = os.path.join(restore_path, f'{net_name}-model.ckpt-{epochs}')
                    saver.restore(sess, ckpt_path)
                except ValueError:
                    ckpt_path = os.path.join(restore_path, f'{net_name}-model-best-val-acc.ckpt-{epochs}')
                    saver.restore(sess, ckpt_path)

                total = int(np.ceil(len(data_dict['x_test']) / batch_size))
                iterator = test_ds.make_initializable_iterator()
                sess.run(iterator.initializer, feed_dict={filenames_placeholder: data_dict['x_test'],
                                                          labels_placeholder: data_dict['y_test']})
                next_element = iterator.get_next()

                y_test = []
                y_pred_all = []
                pbar = tqdm(range(total))
                for i in pbar:
                    batch_x, batch_y = sess.run(next_element)
                    pbar.set_description(f'predicting target present/absent for batch {i} of {total}')
                    feed = {x: batch_x}
                    y_pred = sess.run(predictions['labels'], feed_dict=feed)
                    y_pred_all.append(y_pred)
                    y_test.append(batch_y)
                y_pred_all = np.concatenate(y_pred_all)
                y_test = np.concatenate(y_test)
                set_size_vec_test = data_dict['set_size_vec_test']
                acc_per_set_size = []
                for set_size in set_sizes:
                    # in line below, [0] at end because np.where returns a tuple
                    inds = np.where(set_size_vec_test == set_size)[0]
                    acc_this_set_size = (np.sum(y_pred_all[inds] == y_test[inds]) /
                                         y_test[inds].shape)
                    acc_per_set_size.append(acc_this_set_size)

                acc_set_size_str = ''
                acc_set_size_zip = zip(set_sizes, acc_per_set_size)
                for set_size, acc in acc_set_size_zip:
                    acc_set_size_str += f'set size {set_size}: {acc}. '
                print(acc_set_size_str)

                # append to list of lists which we convert into matrix
                acc_per_set_size_per_model.append(acc_per_set_size)
                # and insert into dictionary where model name is key
                # and list of accuracies per set size is the "value"
                acc_per_set_size_model_dict[restore_path] = acc_per_set_size
                predictions_per_model_dict[restore_path] = y_pred_all

        acc_per_set_size_per_model = np.asarray(acc_per_set_size_per_model)
        acc_per_set_size_per_model = np.squeeze(acc_per_set_size_per_model)

        if not os.path.isdir(test_results_save_path):
            os.makedirs(test_results_save_path)
        results_fname_stem = str(Path(configfile).stem)  # remove .ini extension
        results_fname = os.path.join(test_results_save_path,
                                     f'{results_fname_stem}_trained_{epochs}_epochs_test_results.gz')
        results_dict = dict(acc_per_set_size_per_model=acc_per_set_size_per_model,
                            acc_per_set_size_model_dict=acc_per_set_size_model_dict,
                            predictions_per_model_dict=predictions_per_model_dict,
                            set_sizes=set_sizes)
        joblib.dump(results_dict, results_fname)
