# tensorflow utils
# plagiarized mercilessly from Sebastian Raschka
#https://github.com/rasbt/python-machine-learning-book-2nd-edition
import os

import numpy as np
import tensorflow as tf


def batch_generator(X, y, batch_size=64,
                    shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])

    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]

    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size, :], y[i:i + batch_size])


def save(saver, sess, epoch, path='./model/'):
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Saving model in %s' % path)
    saver.save(sess, os.path.join(path, 'alexnet-model.ckpt'),
               global_step=epoch)


def load(saver, sess, path, epoch):
    print('Loading model from %s' % path)
    saver.restore(sess, os.path.join(
            path, 'alexnet-model.ckpt-%d' % epoch))


def train(sess, training_set, validation_set=None,
          initialize=True, epochs=20, shuffle=True,
          random_seed=None, batch_size=64):

    X_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    training_loss = []

    # initialize variables
    if initialize:
        sess.run(tf.global_variables_initializer())

    np.random.seed(random_seed)  # for shuffling in batch_generator
    for epoch in range(1, epochs+1):
        batch_gen = batch_generator(X_data, y_data,
                                    batch_size=batch_size,
                                    shuffle=shuffle)
        avg_loss = 0.0
        for i, (batch_x,batch_y) in enumerate(batch_gen):
            feed = {'x:0': batch_x,
                    'y:0': batch_y}
            loss, _ = sess.run(
                    ['cross_entropy_loss:0', 'train_op'],
                    feed_dict=feed)
            avg_loss += loss

        training_loss.append(avg_loss / (i+1))
        print('Epoch %02d Training Avg. Loss: %7.3f' % (
            epoch, avg_loss), end=' ')
        if validation_set is not None:
            feed = {'x:0': validation_set[0],
                    'y:0': validation_set[1]}
            valid_acc = sess.run('accuracy:0', feed_dict=feed)
            print(' Validation Acc: %7.3f' % valid_acc)
        else:
            print()


def predict(sess, X_test, return_proba=False):
    feed = {'x:0': X_test}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)