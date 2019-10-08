"""Trainer class"""
import torch
import torch.nn as nn

from .. import nets
from .abstract_trainer import AbstractTrainer
# from .triplet_loss import batch_all_triplet_loss, dist_squared, dist_euclid


class Trainer(AbstractTrainer):
    """class for training CNNs on visual search task.
    Networks are trained 'from scratch', i.e. weights are randomly initialized,
    as opposed to TransferTrainer that uses weights pre-trained on ImageNet"""
    def __init__(self, **kwargs):
        """create new Trainer instance.
        See AbstractTrainer.__init__ docstring for parameters.
        """
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls,
                    net_name,
                    num_classes,
                    loss_func='ce',
                    optimizer='SGD',
                    learning_rate=0.001,
                    momentum=0.9,
                    **kwargs,
                    ):
        """factory function that creates instance of Trainer from options specified in config.ini file

        Parameters
        ----------
        net_name : str
            name of convolutional neural net architecture to train.
            One of {'alexnet', 'VGG16'}
        num_classes : int
            number of classes. Default is 2 (target present, target absent).
        loss_func : str
            type of loss function to use. One of {'CE', 'InvDPrime', 'triplet'}. Default is 'CE',
            the standard cross-entropy loss. 'InvDPrime' is inverse D prime. 'triplet' is triplet loss
            used in face recognition and biometric applications.
        learning_rate : float
            value for learning rate hyperparameter. Default is 0.001 (which is what
            was used to train AlexNet and VGG16).
        momentum : float
            value for momentum hyperparameter of optimizer. Default is 0.9 (which is what
            was used to train AlexNet and VGG16).
        kwargs : dict

        Returns
        -------
        trainer : Trainer
        """
        if net_name == 'alexnet':
            model = nets.alexnet.build(pretrained=False, num_classes=num_classes)
        elif net_name == 'VGG16':
            model = nets.vgg16.build(pretrained=False, num_classes=num_classes)

        optimizers = list()
        if optimizer == 'SGD':
            optimizers.append(
                torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=momentum))
        elif optimizer == 'Adam':
            optimizers.append(
                torch.optim.Adam(model.parameters(),
                                 lr=learning_rate))
        elif optimizer == 'AdamW':
            optimizers.append(
                torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate))

        if loss_func == 'CE':
            criterion = nn.CrossEntropyLoss()
        # elif loss_func == 'triplet':
        #     loss_op, fraction = batch_all_triplet_loss(y, embeddings, margin=triplet_loss_margin,
        #                                                squared=squared_dist)
        # elif loss_func == 'triplet-CE':
        #     CE_loss_op = tf.reduce_mean(
        #         tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.output,
        #                                                    labels=y_onehot),
        #         name='cross_entropy_loss')
        #     triplet_loss_op, fraction = batch_all_triplet_loss(y, embeddings, margin=triplet_loss_margin,
        #                                                        squared=squared_dist)
        #     train_summaries.extend([
        #         tf.summary.scalar('cross_entropy_loss', CE_loss_op),
        #         tf.summary.scalar('triplet_loss', triplet_loss_op),
        #     ])
        #     loss_op = CE_loss_op + triplet_loss_op

        kwargs = dict(**kwargs, net_name=net_name, model=model, optimizers=optimizers, criterion=criterion)
        trainer = cls(**kwargs)
        return trainer
