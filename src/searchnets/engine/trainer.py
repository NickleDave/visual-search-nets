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
                    trainset,
                    mode='classify',
                    optimizer='SGD',
                    embedding_n_out=512,
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
        optimizer : str
            optimizer to use. One of {'SGD', 'Adam', 'AdamW'}.
        embedding_n_out : int
            for DetectNet, number of output features from input embedding.
            I.e., the output size of the linear layer that accepts the
            one hot vector querying whether a specific class is present as input.
            Default is 512.
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
        elif 'cornet' in net_name.lower():
            model = nets.cornet.build(model_name=net_name, pretrained=False,
                                      num_classes=num_classes)
        else:
            raise ValueError(
                f'invalid value for net_name: {net_name}'
            )

        if mode == 'detect':
            # remove final output layer, will replace
            if net_name == 'alexnet' or net_name == 'VGG16':
                model.classifier = model.classifier[:-1]
            elif 'cornet' in net_name.lower():
                # for CORnet models, also need to remove 'output' layer (just an Identity)
                model.decoder = model.decoder[:-2]
            a_sample = next(iter(trainset))
            tmp_img = a_sample['img'].unsqueeze(0)  # add batch dim
            tmp_out = model(tmp_img)
            vis_sys_n_features_out = tmp_out.shape[-1]  # (batch, n features)
            model = nets.detectnet.DetectNet(vis_sys=model,
                                             num_classes=num_classes,
                                             vis_sys_n_out=vis_sys_n_features_out,
                                             embedding_n_out=embedding_n_out)

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

        trainer = cls(net_name=net_name,
                      model=model,
                      optimizers=optimizers,
                      trainset=trainset,
                      mode=mode,
                      **kwargs)
        return trainer
