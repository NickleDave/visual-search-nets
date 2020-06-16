"""TransferTrainer class"""
import torch

from .. import nets
from .abstract_trainer import AbstractTrainer


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
                    trainset,
                    mode='classify',
                    num_classes=2,
                    embedding_n_out=512,
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
        new_learn_rate_layers : list
            of str, layer names whose weights will be initialized randomly
            and then trained with the 'new_layer_learning_rate'.
        trainset : torch.Dataset or torchvision.Visiondataset
            training data, represented as a class.
        mode : str
            training mode. One of {'classify', 'detect'}.
            'classify' is standard image classification.
            'detect' trains to detect whether specified target is present or absent.
            Default is 'classify'.
        num_classes : int
            number of classes. Default is 2 (target present, target absent).
        embedding_n_out : int
            for DetectNet, number of output features from input embedding.
            I.e., the output size of the linear layer that accepts the
            one hot vector querying whether a specific class is present as input.
            Default is 512.
        optimizer : str
            optimizer to use. One of {'SGD', 'Adam', 'AdamW'}.
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
        elif 'cornet' in net_name.lower():
            model = nets.cornet.build(model_name=net_name, pretrained=True)
            model = nets.cornet.reinit(model, model_name=net_name, num_classes=num_classes)
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

        optimizers = []
        if mode == 'classify':
            # violating principle of Don't Repeat Yourself here
            # to be extra careful not to change behavior of previous code;
            # all code in this 'if' block is what was used to assign parameters
            # to optimizers before adding the 'detect' mode
            if net_name == 'alexnet' or net_name == 'VGG16':
                classifier_params = model.classifier.parameters()
            elif 'cornet' in net_name.lower():
                classifier_params = model.decoder.parameters()

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

            if net_name == 'alexnet' or net_name == 'VGG16':
                feature_params = model.features.parameters()
            elif 'cornet' in net_name.lower():
                feature_params = [list(p) for p in
                                  [model.V1.parameters(), model.V2.parameters(),
                                   model.V4.parameters(), model.IT.parameters()]]
                feature_params = [p for params in feature_params for p in params]

            if freeze_trained_weights:
                for params in feature_params:
                    params.requires_grad = False
            else:
                if optimizer == 'SGD':
                    optimizers.append(
                        torch.optim.SGD(feature_params,
                                        lr=base_learning_rate,
                                        momentum=momentum))
                elif optimizer == 'Adam':
                    optimizers.append(
                        torch.optim.Adam(feature_params,
                                         lr=base_learning_rate))
                elif optimizer == 'AdamW':
                    optimizers.append(
                        torch.optim.AdamW(feature_params,
                                          lr=base_learning_rate))
        elif mode == 'detect':
            if net_name == 'alexnet' or net_name == 'VGG16':
                new_learn_rate_params = list(model.vis_sys.classifier.parameters())
            elif 'cornet' in net_name.lower():
                new_learn_rate_params = list(model.vis_sys.decoder.parameters())
            new_learn_rate_params += list(model.embedding.parameters())
            new_learn_rate_params += list(model.decoder.parameters())

            if optimizer == 'SGD':
                optimizers.append(
                    torch.optim.SGD(new_learn_rate_params,
                                    lr=new_layer_learning_rate,
                                    momentum=momentum))
            elif optimizer == 'Adam':
                optimizers.append(
                    torch.optim.Adam(new_learn_rate_params,
                                     lr=new_layer_learning_rate))
            elif optimizer == 'AdamW':
                optimizers.append(
                    torch.optim.AdamW(new_learn_rate_params,
                                      lr=new_layer_learning_rate))

            if net_name == 'alexnet' or net_name == 'VGG16':
                feature_params = model.vis_sys.features.parameters()
            elif 'cornet' in net_name.lower():
                feature_params = [list(p) for p in
                                  [model.vis_sys.V1.parameters(), model.vis_sys.V2.parameters(),
                                   model.vis_sys.V4.parameters(), model.vis_sys.IT.parameters()]]
                feature_params = [p for params in feature_params for p in params]

            if freeze_trained_weights:
                for params in feature_params:
                    params.requires_grad = False
            else:
                if optimizer == 'SGD':
                    optimizers.append(
                        torch.optim.SGD(feature_params,
                                        lr=base_learning_rate,
                                        momentum=momentum))
                elif optimizer == 'Adam':
                    optimizers.append(
                        torch.optim.Adam(feature_params,
                                         lr=base_learning_rate))
                elif optimizer == 'AdamW':
                    optimizers.append(
                        torch.optim.AdamW(feature_params,
                                          lr=base_learning_rate))
        trainer = cls(net_name=net_name,
                      model=model,
                      optimizers=optimizers,
                      trainset=trainset,
                      mode=mode,
                      **kwargs)
        return trainer
