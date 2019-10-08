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
                    optimizer='SGD',
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
            name of convolutional neural net architecture to train.
            One of {'alexnet', 'VGG16'}
        num_classes : int
            number of classes. Default is 2 (target present, target absent).
        new_learn_rate_layers : list
            of str, layer names whose weights will be initialized randomly
        and then trained with the 'new_layer_learning_rate'.
        loss_func : str
            type of loss function to use. One of {'CE', 'InvDPrime', 'triplet'}. Default is 'CE',
            the standard cross-entropy loss. 'InvDPrime' is inverse D prime. 'triplet' is triplet loss
            used in face recognition and biometric applications.
        freeze_trained_weights : bool
            if True, freeze weights in any layer not in "new_learn_rate_layers".
            These are the layers that have weights pre-trained on ImageNet.
            Default is False. Done by simply not applying gradients to these weights,
            i.e. this will ignore a base_learning_rate if you set it to something besides zero.
        base_learning_rate : float
            Applied to layers with weights loaded from training the
            architecture on ImageNet. Should be a very small number
            so the trained weights don't change much.
        new_layer_learning_rate : float
            Applied to `new_learn_rate_layers'. Should be larger than
            `base_learning_rate` but still smaller than the usual
            learning rate for a deep net trained with SGD,
            e.g. 0.001 instead of 0.01
        momentum : float
            value for momentum hyperparameter of optimizer. Default is 0.9.
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

        if optimizer == 'SGD':
            optimizer = torch.optim.SGD
        elif optimizer == 'Adam':
            optimizer = torch.optim.Adam
        elif optimizer == 'AdamW':
            optimizer = torch.optim.AdamW

        optimizers = []
        classifier_params = model.classifier.parameters()

        if optimizer == 'SGD':
            optimizers.append(
                torch.optim.SGD(classifier_params,
                                lr=new_layer_learning_rate,
                                momentum=momentum))
        elif optimizer == 'Adam':
            optimizers.append(
                torch.optim.Adam(classifier_params,
                                 lr=new_layer_learning_rate))
        elif optimizer == 'AdamW':
            optimizers.append(
                torch.optim.AdamW(classifier_params,
                                  lr=new_layer_learning_rate))

        optimizers.append(
            torch.optim.SGD(classifier_params,
                            lr=new_layer_learning_rate,
                            momentum=momentum))
        if freeze_trained_weights:
            for params in model.features.parameters():
                params.requires_grad = False
        else:
            feature_params = model.features.parameters()
            if optimizer == 'SGD':
                optimizers.append(
                    torch.optim.SGD(feature_params,
                                    lr=new_layer_learning_rate,
                                    momentum=momentum))
            elif optimizer == 'Adam':
                optimizers.append(
                    torch.optim.Adam(feature_params,
                                     lr=new_layer_learning_rate))
            elif optimizer == 'AdamW':
                optimizers.append(
                    torch.optim.AdamW(feature_params,
                                      lr=new_layer_learning_rate))

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

        kwargs = dict(**kwargs,
                      net_name=net_name,
                      model=model,
                      optimizers=optimizers,
                      criterion=criterion,
                      )
        trainer = cls(**kwargs)
        return trainer
