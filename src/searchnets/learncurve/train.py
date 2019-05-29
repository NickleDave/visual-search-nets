import os

import numpy as np
import tensorflow as tf
import joblib
from tqdm import tqdm

from searchnets.nets import AlexNet
from searchnets.nets import VGG16
from searchnets.train import batch_generator


def train(x_train, y_train, train_size_list, epochs_list, net_name, number_nets_to_train, input_shape,
          new_learn_rate_layers, batch_size, dropout_rate, model_save_path):
    """train networks for a learning curve"""
    all_train_inds = np.arange(x_train.shape[0])

    for train_size in train_size_list:

        train_inds = np.random.choice(all_train_inds, train_size)
        X_data = np.array(x_train)[train_inds]
        y_data = np.array(y_train)[train_inds]

        for epochs in epochs_list:
            print(
                f'training {net_name} model with {train_size} samples for {epochs} epochs'
            )
            for net_number in range(number_nets_to_train):
                print(f'training replicate {net_number + 1} of {number_nets_to_train}')
                tf.reset_default_graph()
                graph = tf.Graph()
                with tf.Session(graph=graph) as sess:
                    x = tf.placeholder(tf.float32, (None,) + input_shape, name='x')
                    y = tf.placeholder(tf.int32, shape=[None], name='y')
                    y_onehot = tf.one_hot(indices=y, depth=len(np.unique(y_train)),
                                          dtype=tf.float32, name='y_onehot')
                    rate = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), shape=(),
                                                       name='dropout_rate')

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

                    training_loss = []

                    # note that running global_variables_initializer() will initialize at random all the variables in
                    # the model that are in the `init_layer` list passed as an argument when the model was instantiated,
                    #  **and** assign the pre-trained weights + biases to the other variables that are not in
                    # `init_layer`. This can be confusing if you are thinking of "initialize" as synonymous with
                    # "at random", but fear not, the pre-trained weights are in fact being loaded
                    sess.run(tf.global_variables_initializer())

                    for epoch in range(1, epochs + 1):
                        total = int(np.ceil(X_data.shape[0] / batch_size))
                        batch_gen = batch_generator(X_data, y_data,
                                                    batch_size=batch_size,
                                                    shuffle=True)
                        avg_loss = 0.0
                        pbar = tqdm(enumerate(batch_gen), total=total)
                        for i, (batch_x, batch_y) in pbar:
                            pbar.set_description(f'batch {i} of {total}')
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
                        if x_val is not None:
                            batch_gen = batch_generator(x_val, y_val,
                                                        batch_size=batch_size,
                                                        shuffle=False)
                            total = int(np.ceil(x_val.shape[0] / batch_size))
                            pbar = tqdm(enumerate(batch_gen), total=total)

                            valid_acc = []
                            for i, (batch_x, batch_y) in pbar:
                                pbar.set_description(f'batch {i} of {total}')
                                feed = {x: batch_x,
                                        y: batch_y,
                                        rate: dropout_rate}

                                valid_acc.append(sess.run(accuracy, feed_dict=feed))
                            valid_acc = np.asarray(valid_acc).mean()

                            print(' Validation Acc: %7.3f' % valid_acc)
                        else:
                            print()

                    savepath = os.path.join(model_save_path,
                                            f'training_set_with_{train_size}_samples',
                                            f'trained_{epochs}_epochs',
                                            f'net_number_{net_number}')
                    if not os.path.isdir(savepath):
                        os.makedirs(savepath, exist_ok=True)
                    print(f'Saving model in {savepath}')
                    ckpt_name = os.path.join(savepath, f'{net_name}-model.ckpt')
                    saver.save(sess, ckpt_name, global_step=epochs)
                    joblib.dump(train_inds, os.path.join(savepath, 'train_inds'))