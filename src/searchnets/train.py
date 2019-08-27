# Approach to training based on https://arxiv.org/pdf/1707.09775.pdf
import os

import joblib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .nets import AlexNet
from .nets import VGG16
from .triplet_loss import batch_all_triplet_loss, dist_squared, dist_euclid
from .utils.tfdata import get_dataset

MOMENTUM = 0.9  # used for both Alexnet and VGG16


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
          loss_func='CE',
          triplet_loss_margin=0.5,
          squared_dist=False,
          use_val=True,
          val_epoch=None,
          summary_step=None,
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
    loss_func : str
        type of loss function to use. One of {'CE', 'InvDPrime', 'triplet'}. Default is 'CE',
        the standard cross-entropy loss. 'InvDPrime' is inverse D prime. 'triplet' is triplet loss
        used in face recognition and biometric applications.
    triplet_loss_margin : float
        Minimum margin between clusters, parameter in triplet loss function. Default is 0.5.
    squared_dist : bool
        if True, when computing similarity of embeddings (e.g. for triplet loss), use pairwise squared
        distance, i.e. Euclidean distance.
    save_acc_by_set_size_by_epoch : bool
        if True, compute accuracy on training set for each epoch separately
        for each unique set size in the visual search stimuli. These values
        are saved in a matrix where rows are epochs and columns are set sizes.
        Useful for seeing whether accuracy converges for each individual
        set size. Default is False.
    use_val : bool
        if True, use validation set.
    val_epoch : int
        if not None, accuracy on validation set will be measured every `val_epoch` epochs. Default is None.
    summary_step : int
        Step on which to write summaries to file. Each minibatch is counted as one step, and steps are counted across
        epochs. Default is None.
    patience : int
        if not None, training will stop if accuracy on validation set has not improved in `patience` steps

    Returns
    -------
    None
    """
    if use_val and val_epoch is None or val_epoch < 1 or type(val_epoch) != int:
        raise ValueError(
            'invalid value for val_epoch: {val_epoch}. Validation epoch must be positive integer'
        )

    if use_val is False and patience is not None:
        raise ValueError('patience argument only works with a validation set')

    if patience is not None:
        if type(val_epoch) != int or patience < 1:
            raise TypeError('patience must be a positive integer')

    if type(epochs_list) is int:
        epochs_list = [epochs_list]
    elif type(epochs_list) is list:
        pass
    else:
        raise TypeError("'EPOCHS' option in 'TRAIN' section of config.ini file parsed "
                        f"as invalid type: {type(epochs_list)}")

    print('loading training data')
    data_dict = joblib.load(gz_filename)

    if 'shard_train' in data_dict:
        shard_train = data_dict['shard_train']
    else:
        shard_train = False

    np.random.seed(random_seed)  # for shuffling in batch_generator
    tf.random.set_random_seed(random_seed)

    if save_acc_by_set_size_by_epoch:
        # get vecs for computing accuracy by set size below
        # in training loop
        if shard_train:
            set_size_vec_train = np.concatenate(data_dict['set_size_vec_train'])
        else:
            set_size_vec_train = data_dict['set_size_vec_train']

        set_sizes = np.unique(set_size_vec_train)

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
                # apparently it matters if we make the tf.data.Dataset in the same graph as the network
                filenames_placeholder_tr = tf.placeholder(tf.string, shape=[None])
                labels_placeholder_tr = tf.placeholder(tf.int64, shape=[None])
                if shard_train:
                    shuffle_size = data_dict['shard_size']
                else:
                    shuffle_size = len(data_dict['x_train'])
                train_ds = get_dataset(filenames_placeholder_tr, labels_placeholder_tr, net_name, batch_size,
                                       shuffle=True, shuffle_size=shuffle_size)

                if use_val:
                    if 'x_val' not in data_dict:
                        raise KeyError(
                            f'use_val set to True but x_val not found in data file: {gz_filename}'
                        )

                    filenames_placeholder_val = tf.placeholder(tf.string, shape=[None])
                    labels_placeholder_val = tf.placeholder(tf.int64, shape=[None])
                    val_ds = get_dataset(filenames_placeholder_val, labels_placeholder_val, net_name, batch_size,
                                         shuffle=False, shuffle_size=None)
                else:
                    val_ds = None

                if save_acc_by_set_size_by_epoch:
                    filenames_placeholder_acc = tf.placeholder(tf.string, shape=[None])
                    labels_placeholder_acc = tf.placeholder(tf.int64, shape=[None])
                    train_ds_no_shuffle = get_dataset(filenames_placeholder_acc, labels_placeholder_acc,
                                                      net_name, batch_size, shuffle=False, shuffle_size=None)

                x = tf.placeholder(tf.float32, (None,) + input_shape, name='x')
                y = tf.placeholder(tf.int32, shape=[None], name='y')
                if shard_train:
                    depth = len(np.unique(np.concatenate(data_dict['y_train'])))
                else:
                    depth = len(np.unique(data_dict['y_train']))
                y_onehot = tf.one_hot(indices=y, depth=depth, dtype=tf.float32, name='y_onehot')
                rate = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), shape=(), name='dropout_rate')

                if net_name == 'alexnet':
                    model = AlexNet(x, init_layer=new_learn_rate_layers, dropout_rate=rate)
                elif net_name == 'VGG16':
                    model = VGG16(x, init_layer=new_learn_rate_layers, dropout_rate=rate)

                predictions = {
                    'probabilities': tf.nn.softmax(model.output, name='probabilities'),
                    'labels': tf.cast(tf.argmax(model.output, axis=1), tf.int32, name='labels')
                }

                embeddings = model.fc7
                # t = target, d = distractor
                t_inds = tf.where(tf.math.equal(y, 1))
                t_vecs = tf.gather(embeddings, t_inds)
                t_vecs = tf.squeeze(t_vecs)
                if squared_dist:
                    t_distances = dist_squared(t_vecs)
                else:
                    t_distances = dist_euclid(t_vecs)
                tf.summary.histogram('target_distances', t_distances)
                tf.summary.scalar('target_distances_mean', tf.reduce_mean(t_distances))
                tf.summary.scalar('target_distances_std', tf.math.reduce_std(t_distances))

                d_inds = tf.where(tf.math.equal(y, 0))
                d_vecs = tf.gather(embeddings, d_inds)
                d_vecs = tf.squeeze(d_vecs)
                if squared_dist:
                    d_distances = dist_squared(d_vecs)
                else:
                    d_distances = dist_euclid(d_vecs)
                tf.summary.histogram('distractor_distances', d_distances)
                tf.summary.scalar('distractor_distances_mean', tf.reduce_mean(d_distances))
                tf.summary.scalar('distractor_distances_std', tf.math.reduce_std(d_distances))

                if squared_dist:
                    td_distances = dist_squared(t_vecs, d_vecs)
                else:
                    td_distances = dist_euclid(t_vecs, d_vecs)
                tf.summary.histogram('target_distractor_distances', td_distances)
                tf.summary.scalar('target_distractor_distances_mean', tf.reduce_mean(td_distances))
                tf.summary.scalar('target_distractor_distances_std', tf.math.reduce_std(td_distances))

                if loss_func == 'CE':
                    loss_op = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.output,
                                                                   labels=y_onehot),
                        name='cross_entropy_loss')

                elif loss_func == 'InvDPrime':
                    target_present_inds = tf.math.equal(y, tf.constant(1, tf.int32))
                    vecs_present = tf.gather(model.fc7, target_present_inds)
                    mu_present = tf.math.reduce_mean(vecs_present)
                    sigma_present = tf.math.reduce_std(vecs_present)

                    target_absent_inds = tf.math.equal(y, tf.constant(0, tf.int32))
                    vecs_absent = tf.gather(model.fc7, target_absent_inds)
                    mu_absent = tf.math.reduce_mean(vecs_absent)
                    sigma_absent = tf.math.reduce_std(vecs_absent)

                    cross_entropy_op = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.output,
                                                                   labels=y_onehot),
                        name='cross_entropy_loss')
                    loss_op = sigma_present + sigma_absent + 1 / tf.math.abs()
                elif loss_func == 'triplet':
                    loss_op, fraction = batch_all_triplet_loss(y, embeddings, margin=triplet_loss_margin,
                                                               squared=squared_dist)
                elif loss_func == 'triplet-CE':
                    CE_loss_op = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.output,
                                                                   labels=y_onehot),
                        name='cross_entropy_loss')
                    tf.summary.scalar('cross_entropy_loss', CE_loss_op)
                    triplet_loss_op, fraction = batch_all_triplet_loss(y, embeddings, margin=triplet_loss_margin,
                                                                       squared=squared_dist)
                    tf.summary.scalar('triplet_loss', triplet_loss_op)
                    loss_op = CE_loss_op + triplet_loss_op
                tf.summary.scalar('loss', loss_op)

                var_list1 = []  # all layers before fully-connected
                var_list2 = []  # fully-connected layers
                for train_var in tf.trainable_variables():
                    if any([new_rate_name in train_var.name
                            for new_rate_name in new_learn_rate_layers]):
                        var_list2.append(train_var)
                    else:
                        var_list1.append(train_var)

                if freeze_trained_weights:
                    opt = tf.train.MomentumOptimizer(learning_rate=new_layer_learning_rate,
                                                     momentum=MOMENTUM)
                    grads = tf.gradients(loss_op, var_list2)
                    train_op = opt.apply_gradients(zip(grads, var_list2), name='train_op')
                else:
                    opt1 = tf.train.MomentumOptimizer(learning_rate=base_learning_rate,
                                                      momentum=MOMENTUM)
                    opt2 = tf.train.MomentumOptimizer(learning_rate=new_layer_learning_rate,
                                                      momentum=MOMENTUM)
                    grads = tf.gradients(loss_op, var_list1 + var_list2)
                    grads1 = grads[:len(var_list1)]
                    grads2 = grads[len(var_list1):]
                    train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
                    train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
                    train_op = tf.group(train_op1, train_op2, name='train_op')

                correct_predictions = tf.equal(predictions['labels'],
                                               y, name='correct_preds')

                accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, tf.float32),
                    name='accuracy')
                tf.summary.scalar('accuracy', accuracy)

                summaries = tf.summary.merge_all()
                saver = tf.train.Saver()
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

                if summary_step:
                    train_writer = tf.summary.FileWriter(os.path.join(savepath, 'train'),
                                                         sess.graph)
                if save_acc_by_set_size_by_epoch:
                    acc_by_epoch_by_set_size = np.zeros(shape=(epochs, set_sizes.shape[0]))

                # --------------- finally start training ---------------------------------------------------------------
                step = 0  # each minibatch is a step, and we count steps across epochs
                if val_ds is not None:
                    val_acc = []
                    if patience is not None:
                        best_val_acc = 0
                        epochs_without_improvement = 0

                for epoch in range(epochs):
                    print(f'\nEpoch {epoch + 1}')
                    if shard_train:
                        shard_total = len(data_dict['x_train'])
                        shards = zip(data_dict['x_train'], data_dict['y_train'])
                    else:
                        batch_total = int(np.ceil(len(data_dict['x_train']) / batch_size))
                        shard_total = 1
                        # wrap in list to iterate over (a list of length one)
                        shards = zip([data_dict['x_train']], [data_dict['y_train']])
                    total_loss = 0.0
                    shard_pbar = tqdm(range(shard_total))

                    with shard_pbar:
                        for shard_num, (x_shard, y_shard) in enumerate(shards):
                            shard_pbar.update(shard_num)
                            shard_pbar.set_description(f'shard {shard_num} of {shard_total}')
                            iterator = train_ds.make_initializable_iterator()
                            sess.run(iterator.initializer, feed_dict={filenames_placeholder_tr: x_shard,
                                                                      labels_placeholder_tr: y_shard})
                            next_element = iterator.get_next()
                            batch_total = int(np.ceil(len(x_shard) / batch_size))
                            batch_pbar = tqdm(range(batch_total))
                            for i in batch_pbar:
                                step += 1
                                batch_x, batch_y = sess.run(next_element)
                                feed = {x: batch_x,
                                        y: batch_y,
                                        rate: dropout_rate}
                                if summary_step:
                                    if step % summary_step == 0:
                                        summary, loss, _ = sess.run([summaries, loss_op, train_op],
                                                                    feed_dict=feed)
                                        train_writer.add_summary(summary, step)
                                else:
                                    loss, _ = sess.run(
                                        [loss_op, train_op],
                                        feed_dict=feed)
                                batch_pbar.set_description(f'batch {i} of {batch_total}, loss: {loss: 7.3f}')
                                total_loss += loss

                    avg_loss = total_loss / (batch_total * shard_total)
                    print(f'\tTraining Avg. Loss: {avg_loss:7.3f}')

                    if val_ds is not None:
                        if epoch % val_epoch == 0:
                            total = int(np.ceil(len(data_dict['x_val']) / batch_size))
                            iterator = val_ds.make_initializable_iterator()
                            sess.run(iterator.initializer, feed_dict={filenames_placeholder_val: data_dict['x_val'],
                                                                      labels_placeholder_val: data_dict['y_val']})
                            next_element = iterator.get_next()
                            pbar = tqdm(range(total))
                            val_acc_this_epoch = []
                            for i in pbar:
                                pbar.set_description(f'batch {i} of {total}')
                                batch_x, batch_y = sess.run(next_element)
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
                        if shard_train:
                            shard_total = len(data_dict['x_train'])
                            shards = zip(data_dict['x_train'], data_dict['y_train'])
                        else:
                            batch_total = int(np.ceil(len(data_dict['x_train']) / batch_size))
                            shard_total = 1
                            # wrap in list to iterate over (a list of length one)
                            shards = zip([data_dict['x_train']], [data_dict['y_train']])
                        shard_pbar = tqdm(range(shard_total))

                        with shard_pbar:
                            for shard_num, (x_shard, y_shard) in enumerate(shards):
                                shard_pbar.update(shard_num)
                                shard_pbar.set_description(f'shard {shard_num} of {shard_total}')
                                iterator = train_ds_no_shuffle.make_initializable_iterator()
                                sess.run(iterator.initializer, feed_dict={filenames_placeholder_acc: x_shard,
                                                                          labels_placeholder_acc: y_shard})
                                next_element = iterator.get_next()

                                y_pred = []
                                y_true = []

                                pbar = tqdm(range(batch_total))
                                for i in pbar:
                                    pbar.set_description(f'batch {i} of {batch_total}')
                                    batch_x, batch_y = sess.run(next_element)
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
                    ckpt_name = os.path.join(savepath, f'{net_name}-model.ckpt')
                    saver.save(sess, ckpt_name, global_step=epochs)

                stem = f'{net_name}_trained_{epochs}_epochs_number_{net_number}'

                if save_acc_by_set_size_by_epoch:
                    # and save matrix with accuracy by epoch by set size
                    acc_savepath_this_epochs = os.path.join(acc_savepath, f'{stem}.txt')
                    np.savetxt(acc_savepath_this_epochs, acc_by_epoch_by_set_size, delimiter=',')
