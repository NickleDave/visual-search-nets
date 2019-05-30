# Approach to training based on https://arxiv.org/pdf/1707.09775.pdf
import csv
import os

import numpy as np
import tensorflow as tf
import joblib
from tqdm import tqdm

from .nets import AlexNet
from .nets import VGG16

IMAGENET_MEAN = [104.00698793, 116.66876762, 122.67891434]


def batch_generator(X, y, batch_size=64,
                    shuffle=False, random_seed=None):
    """generator that yields batches of training data and labels

    Parameters
    ----------
    X : numpy.ndarray
        training data, e.g., images with dimensions (number of samples, height, width, channels)
    y : numpy.ndarray
        labels for training data, e.g. vector of integers
    batch_size : int
        number of elements in each batch from X and y. Default is 64.
    shuffle : bool
        if True, shuffle data before creating generator. Default is False.
    random_seed : int
        integer to seed random number generator. Default is None.

    Returns
    -------
    batch_x : numpy.ndarray
        subset of X where first dimension is of size batch_size
    batch_y : numpy.ndarray
        subset of y where first dimension is of size batch_size

    Notes
    -----
    adapted from code by Sebastian Raschka under MIT license
    https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt
    """
    idx = np.arange(y.shape[0])

    X_for_batch = np.copy(X)
    y_for_batch = np.copy(y)

    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X_for_batch = X_for_batch[idx]
        y_for_batch = y_for_batch[idx]

    for i in range(0, X.shape[0], batch_size):
        yield (X_for_batch[i:i + batch_size, :], y_for_batch[i:i + batch_size])


def train(gz_filename,
          net_name,
          number_nets_to_train,
          input_shape,
          base_learning_rate,
          freeze_trained_weights,
          new_learn_rate_layers,
          new_layer_learning_rate,
          epochs_list,
          batch_size,
          random_seed,
          model_save_path,
          dropout_rate=0.5,
          use_val=True,
          val_step=None,
          patience=None,
          save_acc_by_set_size_by_epoch=False):
    """train convolutional neural networks to perform visual search task.

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
    base_learning_rate : float
        Applied to layers with weights loaded from training the
        architecture on ImageNet. Should be a very small number
        so the trained weights don't change much.
    freeze_trained_weights : bool
        if True, freeze weights in any layer not in "new_learn_rate_layers".
        These are the layers that have weights pre-trained on ImageNet.
        Default is False. Done by simply not applying gradients to these weights,
        i.e. this will ignore a base_learning_rate if you set it to something besides zero.
    new_learn_rate_layers : list
        of layer names whose weights will be initialized randomly
        and then trained with the 'new_layer_learning_rate'.
    new_layer_learning_rate : float
        Applied to `new_learn_rate_layers'. Should be larger than
        `base_learning_rate` but still smaller than the usual
        learning rate for a deep net trained with SGD,
        e.g. 0.001 instead of 0.01
    epochs_list : list
        of training epochs. Replicates will be trained for each
        value in this list. Can also just be one value, but a list
        is useful if you want to test whether effects depend on
        number of training epochs.
    batch_size : int
        number of samples in a batch of training data
    random_seed : int
        to seed random number generator
    model_save_path : str
        path to directory where model checkpoints should be saved
    dropout_rate : float
        Probability that any unit in a layer will "drop out" during
        a training epoch, as a form of regularization. Default is 0.5.
    save_acc_by_set_size_by_epoch : bool
        if True, compute accuracy on training set for each epoch separately
        for each unique set size in the visual search stimuli. These values
        are saved in a matrix where rows are epochs and columns are set sizes.
        Useful for seeing whether accuracy converges for each individual
        set size. Default is False.
    use_val : bool
        if True, use validation set.
    val_step : int
        if not None, accuracy on validation set will be measured every `val_step` steps
    patience : int
        if not None, training will stop if accuracy on validation set has not improved in `patience` steps

    Returns
    -------
    None
    """
    if use_val and val_step is None or val_step < 1 or type(val_step) != int:
        raise ValueError(
            'invalid value for val_step: {val_step}. Validation step must be positive integer'
        )

    if use_val is False and patience is not None:
        raise ValueError('patience argument only works with a validation set')

    if patience is not None:
        if type(val_step) != int or patience < 1:
            raise TypeError('patience must be a positive integer')

    print('loading training data')
    data_dict = joblib.load(gz_filename)
    x_train = data_dict['x_train']
    # pre-process images
    x_train = np.asarray([img[:, :, [2, 1, 0]] - IMAGENET_MEAN for img in x_train])
    y_train = data_dict['y_train']

    if use_val:
        try:
            x_val = data_dict['x_val']
        except KeyError:
            raise KeyError(
                f'use_val set to True but x_val not found in data file: {gz_filename}'
            )
        x_val = np.asarray([img[:, :, [2, 1, 0]] - IMAGENET_MEAN for img in x_val])
        y_val = data_dict['y_val']
    else:
        x_val = None

    if type(epochs_list) is int:
        epochs_list = [epochs_list]
    elif type(epochs_list) is list:
        pass
    else:
        raise TypeError("'EPOCHS' option in 'TRAIN' section of config.ini file parsed "
                        f"as invalid type: {type(epochs_list)}")

    np.random.seed(random_seed)  # for shuffling in batch_generator
    tf.random.set_random_seed(random_seed)

    if save_acc_by_set_size_by_epoch:
        # get vecs for computing accuracy by set size below
        # in training loop
        set_size_vec_train = data_dict['set_size_vec_train']
        set_sizes = np.unique(set_size_vec_train)
        if x_val is not None:
            set_size_vec_val = data_dict['set_size_vec_val']

        acc_savepath = os.path.join(model_save_path,
                                    f'acc_by_epoch_by_set_size')
        if not os.path.isdir(acc_savepath):
            os.makedirs(acc_savepath, exist_ok=True)

    for epochs in epochs_list:
        print(f'training {net_name} model for {epochs} epochs')
        for net_number in range(number_nets_to_train):
            tf.reset_default_graph()
            graph = tf.Graph()
            with tf.Session(graph=graph) as sess:
                # --------------- do a bunch of graph set-up stuff -----------------------------------------------------
                x = tf.placeholder(tf.float32, (None,) + input_shape, name='x')
                y = tf.placeholder(tf.int32, shape=[None], name='y')
                y_onehot = tf.one_hot(indices=y, depth=len(np.unique(y_train)),
                                      dtype=tf.float32, name='y_onehot')
                rate = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), shape=(), name='dropout_rate')

                if net_name == 'alexnet':
                    model = AlexNet(x, init_layer=new_learn_rate_layers, dropout_rate=rate)
                elif net_name == 'VGG16':
                    model = VGG16(x, init_layer=new_learn_rate_layers, dropout_rate=rate)

                predictions = {
                    'probabilities': tf.nn.softmax(model.output, name='probabilities'),
                    'labels': tf.cast(tf.argmax(model.output, axis=1), tf.int32, name='labels')
                }

                cross_entropy_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.output,
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

                if freeze_trained_weights:
                    opt = tf.train.GradientDescentOptimizer(new_layer_learning_rate)
                    grads = tf.gradients(cross_entropy_loss, var_list2)
                    train_op = opt.apply_gradients(zip(grads, var_list2), name='train_op')
                else:
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

                # note that running global_variables_initializer() will initialize at random all the variables in the
                # model that are in the `init_layer` list passed as an argument when the model was instantiated, **and**
                # assign the pre-trained weights + biases to the other variables that are not in `init_layer`. This can
                # be confusing if you are thinking of "initialize" as synonymous with "at random", but fear not, the
                # pre-trained weights are in fact being loaded
                sess.run(tf.global_variables_initializer())

                # --------------- make places to save checkpoints + accuracy -------------------------------------------
                savepath = os.path.join(model_save_path,
                                        f'trained_{epochs}_epochs',
                                        f'net_number_{net_number}')
                if not os.path.isdir(savepath):
                    os.makedirs(savepath, exist_ok=True)

                if save_acc_by_set_size_by_epoch:
                    acc_by_epoch_by_set_size = np.zeros(shape=(epochs, set_sizes.shape[0]))

                # --------------- finally start training ---------------------------------------------------------------
                train_loss = []
                train_acc = []
                if x_val is not None:
                    val_acc = []
                    if patience is not None:
                        best_val_acc = 0
                        epochs_without_improvement = 0

                for epoch in range(epochs):
                    total = int(np.ceil(x_train.shape[0] / batch_size))
                    batch_gen = batch_generator(x_train, y_train,
                                                batch_size=batch_size,
                                                shuffle=True)
                    total_loss = 0.0
                    pbar = tqdm(enumerate(batch_gen), total=total)
                    for i, (batch_x, batch_y) in pbar:
                        feed = {x: batch_x,
                                y: batch_y,
                                rate: dropout_rate}

                        loss, _ = sess.run(
                            [cross_entropy_loss, train_op],
                            feed_dict=feed)
                        pbar.set_description(f'batch {i} of {total}, loss: {loss: 7.3f}')
                        total_loss += loss

                    avg_loss = np.mean(total_loss)
                    train_loss.append(avg_loss)
                    print(f'\nEpoch {epoch + 1}, Training Avg. Loss: {avg_loss:7.3f}')

                    if x_val is not None:
                        if epoch % val_step == 0:
                            batch_gen = batch_generator(x_val, y_val,
                                                        batch_size=batch_size,
                                                        shuffle=False)
                            total = int(np.ceil(x_val.shape[0] / batch_size))
                            pbar = tqdm(enumerate(batch_gen), total=total)

                            val_acc_this_epoch = []
                            for i, (batch_x, batch_y) in pbar:
                                pbar.set_description(f'batch {i} of {total}')
                                feed = {x: batch_x,
                                        y: batch_y,
                                        rate: dropout_rate}

                                val_acc_this_epoch.append(sess.run(accuracy, feed_dict=feed))
                            val_acc_this_epoch = np.asarray(val_acc_this_epoch).mean()
                            val_acc.append(val_acc_this_epoch)

                            print(' Validation Acc: %7.3f' % val_acc_this_epoch)

                            if patience is not None:
                                if val_acc_this_epoch > best_val_acc:
                                    best_val_acc = val_acc_this_epoch
                                    epochs_without_improvement = 0
                                    print(f'Saving model in {savepath}')
                                    ckpt_name = os.path.join(savepath, f'{net_name}-model-best-val-acc.ckpt')
                                    saver.save(sess, ckpt_name, global_step=epochs)

                                else:
                                    epochs_without_improvement += 1
                                    if epochs_without_improvement > patience:
                                        print(
                                            f'greater than {patience} epochs without improvement in validation '
                                            'accuracy, stopping training')

                                        break

                        else:
                            val_acc.append(None)

                    if save_acc_by_set_size_by_epoch:
                        # --- compute accuracy on whole training set, by set size, for this epoch
                        print('Computing accuracy per visual search stimulus set size on training set')
                        total = int(np.ceil(x_train.shape[0] / batch_size))
                        y_pred = []
                        y_true = []
                        batch_gen = batch_generator(x_train, y_train,
                                                    batch_size=batch_size,
                                                    shuffle=False)
                        pbar = tqdm(enumerate(batch_gen), total=total)
                        for i, (batch_x, batch_y) in pbar:
                            pbar.set_description(f'batch {i} of {total}')
                            y_true.append(batch_y)
                            feed = {x: batch_x, rate: 1.0}
                            batch_y_pred = sess.run(predictions['labels'], feed_dict=feed)
                            y_pred.append(batch_y_pred)

                        y_pred = np.concatenate(y_pred)
                        y_true = np.concatenate(y_true)
                        is_correct = np.equal(y_true, y_pred)

                        for set_size_ind, set_size in enumerate(set_sizes):
                            set_size_inds = np.where(set_size_vec_train == set_size)[0]
                            is_correct_set_size = is_correct[set_size_inds]
                            acc_this_set_size = np.sum(is_correct_set_size) / is_correct_set_size.shape[0]
                            acc_by_epoch_by_set_size[epoch, set_size_ind] = acc_this_set_size

                        acc_set_size_str = ''
                        acc_set_size_zip = zip(set_sizes, acc_by_epoch_by_set_size[epoch, :])
                        for set_size, acc in acc_set_size_zip:
                            acc_set_size_str += f'set size {set_size}: {acc}. '
                        print(acc_set_size_str)

                # --------------- done training, save checkpoint + training history info -------------------------------
                if patience is None:
                    # only save at end if we haven't already been saving checkpoints
                    print(f'Saving model in {savepath}')
                    ckpt_name = os.path.join(savepath, f'{net_name}-model-epoch-{epoch}.ckpt')
                    saver.save(sess, ckpt_name, global_step=epochs)

                stem = f'{net_name}_trained_{epochs}_epochs_number_{net_number}'

                # make rows for csv file with training history
                fieldnames = ['train_loss', 'train_acc']
                ziprows = [train_loss, train_acc]
                if x_val is not None:
                    fieldnames.append('val_acc')
                    ziprows.append(val_acc)
                # use * operator to unpack so we don't have to know how many elements are in ziprows
                rows = [list(row) for row in zip(*ziprows)]

                csv_fname = os.path.join(model_save_path, f'{stem}.training_history.csv')
                with open(csv_fname, 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(dict(zip(fieldnames, row)))

                if save_acc_by_set_size_by_epoch:
                    # and save matrix with accuracy by epoch by set size
                    acc_savepath_this_epochs = os.path.join(acc_savepath, f'{stem}.txt')
                    np.savetxt(acc_savepath_this_epochs, acc_by_epoch_by_set_size, delimiter=',')
