"""TransferTrainer class"""
import torch
import torch.nn as nn

from .. import nets
from .abstract_trainer import AbstractTrainer
# from .triplet_loss import batch_all_triplet_loss, dist_squared, dist_euclid


class TransferTrainer(AbstractTrainer):
    """class for training CNNs on visual search task,
    using a transfer learning approach with weights pre-trained on ImageNet"""
    def __init__(self, **kwargs):
        """create new TransferTrainer instance.
        See AbstractTrainer.__init__ docstring for parameters.
        """
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls,
                    net_name,
                    new_learn_rate_layers,
                    num_classes=2,
                    loss_func='ce',
                    freeze_trained_weights=False,
                    base_learning_rate=1e-20,
                    new_layer_learning_rate=0.00001,
                    momentum=0.9,
                    **kwargs,
                    ):
        """factory function that creates instance of TransferTrainer from options specified in config.ini file

        Parameters
        ----------
        net_name : str
        num_classes : int
        new_learn_rate_layers : list
            of str
        loss_func : str
        freeze_trained_weights : bool
        base_learning_rate : float
        new_layer_learning_rate : float
        momentum : float
        kwargs : dict

        Returns
        -------
        trainer : TransferTrainer
        """
        if net_name == 'alexnet':
            model = nets.alexnet.build(pretrained=True, progress=True)
            model = nets.alexnet.reinit(model, new_learn_rate_layers, num_classes=num_classes)
        elif net_name == 'VGG16':
            model = nets.vgg16.build(pretrained=True, progress=True)
            model = nets.vgg16.reinit(model, new_learn_rate_layers, num_classes=num_classes)

        optimizers = []
        classifier_params = model.classifier.parameters()
        if freeze_trained_weights:
            optimizers.append(
                torch.optim.SGD(classifier_params,
                                lr=new_layer_learning_rate,
                                momentum=momentum))
            for params in model.features.parameters():
                params.requires_grad = False
        else:
            optimizers.append(
                torch.optim.SGD(classifier_params,
                                lr=new_layer_learning_rate,
                                momentum=momentum)
            )
            feature_params = model.features.parameters()
            optimizers.append(
                torch.optim.SGD(feature_params,
                                lr=base_learning_rate,
                                momentum=momentum)
            )

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

        kwargs = dict(**kwargs, model=model, optimizers=optimizers, criterion=criterion)
        trainer = cls(**kwargs)
        return trainer