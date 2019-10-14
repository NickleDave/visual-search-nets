"""Tester class"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .. import nets
from ..datasets import Searchstims, VOCDetection
from ..utils.transforms import normalize

NUM_WORKERS = 4


class Tester:
    """class for measuring accuracy of CNNs on test set after training for visual search task"""
    def __init__(self,
                 net_name,
                 model,
                 testset,
                 restore_path,
                 batch_size=64,
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
        testset : torch.Dataset or torchvision.Visiondataset
            test data, represented as a class.
        restore_path : str
            path to directory where checkpoints and train models were saved
        batch_size : int
            number of training samples per batch
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
        model_file = str(restore_path) + '-model.pt'
        model.load_state_dict(
            torch.load(model_file)
        )
        model.to(device)
        self.model = model
        self.device = device

        self.testset = testset
        self.test_loader = DataLoader(self.testset, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers,
                                      pin_memory=True)

        self.batch_size = batch_size

    @classmethod
    def from_config(cls,
                    net_name,
                    num_classes,
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
        elif net_name == 'CORnet_Z':
            model = nets.cornet.build(pretrained=False, num_classes=num_classes)

        kwargs = dict(**kwargs, net_name=net_name, model=model)
        return cls(**kwargs)

    def test(self):
        """method to test trained model

        Returns
        -------
        acc : float
            accuracy on test set
        pred : numpy.ndarray
            predictions for test set
        """
        self.model.eval()

        total = int(np.ceil(len(self.testset) / self.batch_size))
        pbar = tqdm(self.test_loader)
        acc = []
        pred = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pbar):
                pbar.set_description(f'batch {i} of {total}')
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                # below, _ because torch.max returns (values, indices)
                _, pred_batch = torch.max(output.data, 1)
                if batch_y.size(1) > 1:
                    _, batch_y_class = torch.max(batch_y, 1)
                    acc_batch = (pred_batch == batch_y_class).sum().item() / batch_y_class.size(0)
                else:
                    acc_batch = (pred_batch == batch_y).sum().item() / batch_y.size(0)

                acc.append(acc_batch)

                pred_batch = pred_batch.cpu().numpy()
                pred.append(pred_batch)

        acc = np.asarray(acc).mean()
        pred = np.concatenate(pred)

        return acc, pred
