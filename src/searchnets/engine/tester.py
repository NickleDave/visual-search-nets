"""Tester class"""
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import nets
from .. import datasets

from .abstract_trainer import AbstractTrainer

NUM_WORKERS = 4


class Tester:
    """class for measuring accuracy of CNNs on test set after training for visual search task"""
    def __init__(self,
                 net_name,
                 model,
                 criterion,
                 loss_func,
                 testset,
                 restore_path,
                 batch_size=64,
                 sigmoid_threshold=0.5,
                 device='cuda',
                 num_workers=NUM_WORKERS,
                 data_parallel=False,
                 ):
        """create new Tester instance

        Parameters
        ----------
        net_name : str
            name of convolutional neural net architecture to train.
            One of {'alexnet', 'VGG16'}
        model : torch.nn.Module
            actual instance of network.
        criterion : torch.nn._Loss subclass
            used to compute loss on test set
        loss_func : str
            that represents loss function and target that should be used with it.
            Used to determine targets for computing loss, and for metrics to use
            when determining whether to stop early due to metrics computed on
            validation set.
        testset : torch.Dataset or torchvision.Visiondataset
            test data, represented as a class.
        restore_path : Path
            path to directory where checkpoints and train models were saved
        batch_size : int
            number of training samples per batch
        sigmoid_threshold : float
            threshold to use when converting sigmoid outputs to binary vectors.
            Only used for VSD dataset, where multi-label outputs are expected.
        device : str
            One of {'cpu', 'cuda'}
        num_workers : int
            Number of workers used when loading data in parallel. Default is 4.
        data_parallel : bool
            if True, use torch.nn.dataparallel to train network on multiple GPUs. Default is False.
        """
        self.net_name = net_name

        self.data_parallel = data_parallel
        if data_parallel:
            model = nn.DataParallel(model)

        self.restore_path = restore_path

        best_ckpt_path = restore_path.parent.joinpath(
            restore_path.name + AbstractTrainer.BEST_VAL_ACC_CKPT_SUFFIX
        )
        if not best_ckpt_path.exists():
            ckpt_path = restore_path.parent.joinpath(
                restore_path.name + AbstractTrainer.DEFAULT_CKPT_SUFFIX)
            if not ckpt_path.exists():
                raise ValueError(
                    f'did not find a checkpoint file in restore path: {restore_path}.\n'
                    f'Looked for a checkpoint saved upon best val accuracy: {best_ckpt_path.name} \n'
                    f'and for a checkpoint saved during or at the end of training: {ckpt_path.name}'
                )
            self.ckpt_path_loaded_from = ckpt_path
        else:
            self.ckpt_path_loaded_from = best_ckpt_path

        checkpoint = torch.load(self.ckpt_path_loaded_from)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        self.model = model
        self.device = device

        self.criterion = criterion
        self.loss_func = loss_func

        self.testset = testset
        self.test_loader = DataLoader(self.testset, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers,
                                      pin_memory=True)

        self.batch_size = batch_size

        self.sigmoid_threshold = sigmoid_threshold

        if isinstance(self.testset, datasets.VOCDetection):
            self.sigmoid_activation = torch.nn.Sigmoid()
        else:
            self.sigmoid_activation = None

    @classmethod
    def from_config(cls,
                    net_name,
                    num_classes,
                    loss_func,
                    **kwargs):
        """factory function that creates instance of Tester from options specified in config.ini file
        
        Parameters
        ----------
        net_name : str
            name of neural network architecture. Used when restoring model, checkpoints, etc.
        kwargs : keyword arguments

        Returns
        -------
        tester : Tester
            instance of class, initialized with passed attributes.
        """
        if net_name == 'alexnet':
            model = nets.alexnet.build(pretrained=False, progress=False, num_classes=num_classes)
        elif net_name == 'VGG16':
            model = nets.vgg16.build(pretrained=False, progress=False, num_classes=num_classes)
        elif 'cornet' in net_name.lower():
            model = nets.cornet.build(model_name=net_name, pretrained=False, num_classes=num_classes)
        else:
            raise ValueError(
                f'invalid value for net_name: {net_name}'
            )

        if loss_func in {'CE', 'CE-largest', 'CE-random'}:
            criterion = nn.CrossEntropyLoss()
        elif loss_func == 'BCE':
            criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(
                f'invalid value for loss function: {loss_func}'
            )

        kwargs = dict(**kwargs,
                      net_name=net_name,
                      model=model,
                      loss_func=loss_func,
                      criterion=criterion)
        return cls(**kwargs)

    def test(self):
        """method to test trained model

        Returns
        -------
        test_results : dict
        """
        self.model.eval()

        total = int(np.ceil(len(self.testset) / self.batch_size))
        pbar = tqdm(self.test_loader)

        test_loss = []
        pred = []
        if isinstance(self.test_loader.dataset, datasets.Searchstims):
            test_acc = []
            img_names = None
        elif isinstance(self.test_loader.dataset, datasets.VOCDetection):
            test_f1 = []
            test_acc_largest = []
            test_acc_random = []
            img_names = []

        with torch.no_grad():
            for i, batch in enumerate(pbar):
                pbar.set_description(f'batch {i} of {total}')
                batch_x = batch['img'].to(self.device)
                if self.loss_func == 'BCE' or self.loss_func == 'CE':
                    batch_y = batch['target'].to(self.device)
                elif self.loss_func == 'CE-largest':
                    batch_y = batch['largest'].to(self.device)
                elif self.loss_func == 'CE-random':
                    batch_y = batch['random'].to(self.device)

                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                test_loss.append(loss.mean().cpu())  # mean needed for multiple GPUs

                # below, _ because torch.max returns (values, indices)
                _, pred_max = torch.max(output.data, 1)
                if isinstance(self.test_loader.dataset, datasets.Searchstims):
                    acc = (pred_max == batch_y).sum().item() / batch_y.size(0)
                    test_acc.append(acc)
                    pred_batch = pred_max.cpu().numpy()
                    pred.append(pred_batch)
                elif isinstance(self.test_loader.dataset, datasets.VOCDetection):
                    batch_y_onehot = batch['target'].to(self.device)
                    out_sig = self.sigmoid_activation(output)
                    pred_sig = (out_sig > self.sigmoid_threshold).float()
                    f1 = sklearn.metrics.f1_score(batch_y_onehot.cpu().numpy(), pred_sig.cpu().numpy(),
                                                  average='macro')
                    test_f1.append(f1)

                    batch_largest = batch['largest'].to(self.device)
                    acc_largest = (pred_max == batch_largest).sum().item() / batch_largest.size(0)
                    test_acc_largest.append(acc_largest)

                    batch_random = batch['random'].to(self.device)
                    acc_random = (pred_max == batch_random).sum().item() / batch_random.size(0)
                    test_acc_random.append(acc_random)

                    if self.loss_func == 'BCE':
                        pred_batch = pred_sig.cpu().numpy()
                    else:
                        pred_batch = pred_max.cpu().numpy()
                    pred.append(pred_batch)

                    img_names.extend(batch['name'])  # will be a list

        test_results = {
            'loss': np.asarray(test_loss).mean(),
            'pred': np.concatenate(pred)
        }
        if isinstance(self.test_loader.dataset, datasets.Searchstims):
            test_results['acc'] = np.asarray(test_acc).mean()
        elif isinstance(self.test_loader.dataset, datasets.VOCDetection):
            test_results['f1'] = np.asarray(test_f1).mean()
            test_results['acc_largest'] = np.asarray(test_acc_largest).mean()
            test_results['acc_random'] = np.asarray(test_acc_random).mean()
            test_results['img_names'] = img_names

        return test_results
