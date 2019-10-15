"""Trainer class"""
import torch

from .. import nets
from .abstract_trainer import AbstractTrainer


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
                    apply_sigmoid=False,
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
        apply_sigmoid : bool
            if True, apply sigmoid to output of last layer. Default is False.
            Used for multi-label prediction.
        optimizer : str
            optimizer to use. One of {'SGD', 'Adam', 'AdamW'}.
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
            model = nets.alexnet.build(pretrained=False, num_classes=num_classes, apply_sigmoid=apply_sigmoid)
        elif net_name == 'VGG16':
            model = nets.vgg16.build(pretrained=False, num_classes=num_classes, apply_sigmoid=apply_sigmoid)
        elif net_name == 'CORnet_Z':
            model = nets.cornet.build(pretrained=False, num_classes=num_classes, apply_sigmoid=apply_sigmoid)

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

        kwargs = dict(**kwargs, net_name=net_name, model=model, optimizers=optimizers)
        trainer = cls(**kwargs)
        return trainer
