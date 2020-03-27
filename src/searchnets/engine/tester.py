"""Tester class"""
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import nets
from ..datasets import VOCDetection

from .abstract_trainer import AbstractTrainer

NUM_WORKERS = 4


class Tester:
    """class for measuring accuracy of CNNs on test set after training for visual search task"""
    def __init__(self,
                 net_name,
                 model,
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

        self.testset = testset
        self.test_loader = DataLoader(self.testset, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers,
                                      pin_memory=True)

        self.batch_size = batch_size

        self.sigmoid_threshold = sigmoid_threshold

        if type(self.testset) == VOCDetection:
            self.sigmoid_activation = torch.nn.Sigmoid()
        else:
            self.sigmoid_activation = None

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
        if type(self.testset) == VOCDetection:
            img_names = []
        else:
            img_names = None

        with torch.no_grad():
            for i, sample in enumerate(pbar):
                if img_names is not None:
                    batch_x, batch_y, batch_img_name = sample
                else:
                    batch_x, batch_y = sample
                pbar.set_description(f'batch {i} of {total}')
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                if type(self.testset) == VOCDetection:
                    if batch_y.dim() > 1:
                        # convert to one hot vector
                        output = self.sigmoid_activation(output)
                        pred_batch = (output > self.sigmoid_threshold).float()
                    else:
                        raise ValueError(
                            'output of network for VOCDetection dataset only had one dimension, '
                            'should have more than one'
                        )
                    acc_batch = sklearn.metrics.f1_score(batch_y.cpu().numpy(), pred_batch.cpu().numpy(),
                                                         average='macro')
                else:
                    # below, _ because torch.max returns (values, indices)
                    _, pred_batch = torch.max(output.data, 1)
                    acc_batch = (pred_batch == batch_y).sum().item() / batch_y.size(0)

                acc.append(acc_batch)

                pred_batch = pred_batch.cpu().numpy()
                pred.append(pred_batch)
                if img_names is not None:
                    img_names.extend(batch_img_name)

        acc = np.asarray(acc).mean()
        pred = np.concatenate(pred)

        if img_names is not None:
            return acc, pred, img_names
        else:
            return acc, pred
