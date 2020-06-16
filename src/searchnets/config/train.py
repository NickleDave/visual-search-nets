"""class to represent train section of config.ini file """
from pathlib import Path

import attr
from attr import validators
from attr.validators import instance_of

from ..utils.general import projroot_path


def is_pos_int(instance, attribute, value):
    if type(value) != int:
        raise ValueError(
            f'type of {attribute.name} must be an int'
        )
    if value < 1:
        raise ValueError(
            f'{attribute.name} must be a positive integer, but was: {value}'
        )


def is_non_neg_int(instance, attribute, value):
    if type(value) != int:
        raise ValueError(
            f'type of {attribute.name} must be an int'
        )
    if value < 0:
        raise ValueError(
            f'{attribute.name} must be a non-negative integer, but was: {value}'
        )


VALID_LOSS_FUNCTIONS = {
    'BCE',
    'CE',
    'CE-largest',
    'CE-random'
}


@attr.s
class TrainConfig:
    """class to represent [TRAIN] section of config.ini file

    Attributes
    ----------
    net_name : str
        name of convolutional neural net architecture to train.
        One of {'alexnet', 'VGG16'}
    number_nets_to_train : int
        number of training "replicates"
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
    save_path : str
        path to directory where model and any checkpoints should be saved
    method : str
        training method. One of {'initialize', 'transfer'}.
        'initialize' means randomly initialize all weights and train the
        networks "from scratch".
        'transfer' means perform transfer learning, using weights pre-trained
        on imagenet.
        Default is 'transfer'.
    mode : str
        training mode. One of {'classify', 'detect'}.
        'classify' is standard image classification.
        'detect' trains to detect whether specified target is present or absent.
        Default is 'classify'.
    base_learning_rate : float
        Applied to layers with weights loaded from training the
        architecture on ImageNet. Should be a very small number
        so the trained weights don't change much. Default is 0.0
    freeze_trained_weights : bool
        if True, freeze weights in any layer not in "new_learn_rate_layers".
        These are the layers that have weights pre-trained on ImageNet.
        Default is False. Done by simply not applying gradients to these weights,
        i.e. this will ignore a base_learning_rate if you set it to something besides zero.
    dropout_rate : float
        Probability that any unit in a layer will "drop out" during
        a training epoch, as a form of regularization. Default is 0.5.
    embedding_n_out : int
        for DetectNet, number of output features from input embedding.
        I.e., the output size of the linear layer that accepts the
        one hot vector querying whether a specific class is present as input.
        Default is 512.
    loss_func : str
        type of loss function to use. One of {'CE', 'invDPrime'}. Default is 'CE',
        the standard cross-entropy loss. 'invDprime' is inverse D prime.
    optimizer : str
        optimizer to use. One of {'SGD', 'Adam', 'AdamW'}.
    save_acc_by_set_size_by_epoch : bool
        if True, compute accuracy on training set for each epoch separately
        for each unique set size in the visual search stimuli. These values
        are saved in a matrix where rows are epochs and columns are set sizes.
        Useful for seeing whether accuracy converges for each individual
        set size. Default is False.
    use_val : bool
        if True, use validation set.
    val_step : int
        if not None, accuracy on validation set will be measured every `val_step` steps.
        Each minibatch is counted as one step, and steps are counted across
        epochs.
    summary_step : int
        Step on which to write summaries to file.
        Each minibatch is counted as one step, and steps are counted across
        epochs. Default is None.
    patience : int
        if not None, training will stop if accuracy on validation set has not improved in `patience` steps
    ckpt_step : int
        if not None, a checkpoint will be saved every `ckpt_step` steps.
        Each minibatch is counted as one step, and steps are counted across
        epochs. Default is None.
    num_workers : int
        number of workers used by torch.DataLoaders. Default is 4.
    data_parallel : bool
        if True, use torch.nn.DataParallel to train model across multiple GPUs. Default is False.
    """
    net_name = attr.ib(validator=instance_of(str))
    @net_name.validator
    def check_net_name(self, attribute, value):
        if value not in {'alexnet', 'VGG16', 'CORnet_Z', 'CORnet_RT', 'CORnet_S'}:
            raise ValueError(
                f"net_name must be one of {{'alexnet', 'VGG16', 'CORnet_Z', 'CORnet_RT', 'CORnet_S'}}, but was {value}."
            )
    number_nets_to_train = attr.ib(validator=instance_of(int))
    epochs_list = attr.ib(validator=instance_of(list))
    @epochs_list.validator
    def check_epochs_list(self, attribute, value):
        for ind, epochs in enumerate(value):
            if type(epochs) != int:
                raise TypeError('all values in epochs_list should be int but '
                                f'got type {type(epochs)} for element {ind}')

    batch_size = attr.ib(validator=instance_of(int))
    random_seed = attr.ib(validator=instance_of(int))
    save_path = attr.ib(converter=projroot_path,
                        validator=instance_of(Path))

    # ------------------------ have defaults ------------------------------------------------
    method = attr.ib(validator=instance_of(str), default='transfer')
    @method.validator
    def check_method(self, attribute, value):
        if value not in {'initialize', 'transfer'}:
            raise ValueError(
                f"method must be one of {{'initialize', 'transfer'}}, but was {value}."
            )
    mode = attr.ib(validator=instance_of(str), default='classify')
    @mode.validator
    def check_method(self, attribute, value):
        if value not in {'classify', 'detect'}:
            raise ValueError(
                f"method must be one of {{'classify', 'detect'}}, but was {value}."
            )

    # for 'initialize' training
    learning_rate = attr.ib(validator=instance_of(float), default=0.001)

    # for 'transfer' training
    new_learn_rate_layers = attr.ib(validator=instance_of(list), default=['fc8'])
    @new_learn_rate_layers.validator
    def check_new_learn_rate_layers(self, attribute, value):
        for layer_name in value:
            if type(layer_name) != str:
                raise TypeError(f'new_learn_rate_layer names should be strings but got {layer_name}')
    new_layer_learning_rate = attr.ib(validator=instance_of(float), default=0.001)
    base_learning_rate = attr.ib(validator=instance_of(float), default=1e-20)
    freeze_trained_weights = attr.ib(validator=instance_of(bool), default=True)

    embedding_n_out = attr.ib(validator=validators.optional(is_non_neg_int), default=512)

    loss_func = attr.ib(validator=instance_of(str), default='CE')
    @loss_func.validator
    def check_loss_func(self, attribute, value):
        if value not in VALID_LOSS_FUNCTIONS:
            raise ValueError(
                f"loss_func must be one of {VALID_LOSS_FUNCTIONS}, but was {value}."
            )
    optimizer = attr.ib(validator=instance_of(str), default='SGD')
    @optimizer.validator
    def check_optimizer(self, attribute, value):
        if value not in {'SGD', 'Adam', 'AdamW'}:
            raise ValueError(
                f"optimizer must be one of {{'SGD', 'Adam', 'AdamW'}}, but was {value}."
            )

    save_acc_by_set_size_by_epoch = attr.ib(validator=instance_of(bool), default=False)
    use_val = attr.ib(validator=instance_of(bool), default=False)
    val_step = attr.ib(validator=validators.optional(is_pos_int), default=None)
    summary_step = attr.ib(validator=validators.optional(is_pos_int), default=None)
    patience = attr.ib(validator=validators.optional(is_pos_int), default=None)
    ckpt_step = attr.ib(validator=validators.optional(is_pos_int), default=None)
    num_workers = attr.ib(validator=validators.optional(is_non_neg_int), default=4)
    data_parallel = attr.ib(validator=validators.optional(instance_of(bool)), default=False)
