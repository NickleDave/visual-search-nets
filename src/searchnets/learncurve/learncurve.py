import numpy as np
import tensorflow as tf
import joblib

from .train import train
from .test import test


def learning_curve(gz_filename,
                   net_name,
                   number_nets_to_train,
                   input_shape,
                   base_learning_rate,
                   freeze_trained_weights,
                   new_learn_rate_layers,
                   new_layer_learning_rate,
                   epochs_list,
                   train_size_list,
                   batch_size,
                   random_seed,
                   model_save_path,
                   test_results_save_path,
                   dropout_rate=0.5):
    """generate a learning curve, training convolutional neural networks
    to perform visual search task with a range of sizes of training set.

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
    train_size_list : list
        of number of samples in training set. Used to generate a learning curve
        (where x axis is size of training set and y axis is accuracy)
    batch_size : int
        number of samples in a batch of training data
    random_seed : int
        to seed random number generator
    model_save_path : str
        path to directory where model checkpoints should be saved
    test_results_save_path : str
        path to directory where results from measuring accuracy on test set should be saved
    dropout_rate : float
        Probability that any unit in a layer will "drop out" during
        a training epoch, as a form of regularization. Default is 0.5.

    Returns
    -------
    None
    """
    # get training data
    data_dict = joblib.load(gz_filename)
    x_train = data_dict['x_train']
    y_train = data_dict['y_train']

    if type(epochs_list) is int:
        epochs_list = [epochs_list]
    elif type(epochs_list) is list:
        pass
    else:
        raise TypeError("'EPOCHS' option in 'TRAIN' section of config.ini file parsed "
                        f"as invalid type: {type(epochs_list)}")

    np.random.seed(random_seed)  # for shuffling in batch_generator
    tf.random.set_random_seed(random_seed)

    x_test = data_dict['x_test']
    y_test = data_dict['y_test']
    # make sure there's only one 'set' of set sizes
    set_sizes_by_stim_type = data_dict['set_sizes_by_stim_stype']
    set_sizes = []
    for stim_type, set_sizes_this_stim in set_sizes_by_stim_type.items():
        if set_sizes == []:
            set_sizes = set_sizes_this_stim
        else:
            if set_sizes_this_stim != set_sizes:
                raise ValueError('set sizes are not the same across visual search stimuli')
            else:
                continue

    # -------------------- first train a bunch of models ---------------------------------------------------------------
    train(x_train,
          y_train,
          net_name,
          number_nets_to_train,
          input_shape,
          base_learning_rate,
          freeze_trained_weights,
          new_learn_rate_layers,
          new_layer_learning_rate,
          train_size_list,
          epochs_list,
          batch_size,
          dropout_rate,
          model_save_path)

    # -------------------- then measure accuracy on test data ----------------------------------------------------------
    test(x_train,
         y_train,
         x_test,
         y_test,
         data_dict['set_size_vec_train'],
         data_dict['set_size_vec_test'],
         set_sizes,
         net_name,
         number_nets_to_train,
         input_shape,
         new_learn_rate_layers,
         epochs_list,
         train_size_list,
         batch_size,
         model_save_path,
         test_results_save_path)
