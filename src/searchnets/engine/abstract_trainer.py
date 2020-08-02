"""Trainer class"""
from pathlib import Path

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .. import datasets
from ..transforms.functional import tile


class AbstractTrainer:
    """abstract class for training CNNs on visual search task.
    Both Trainer and TransferTrainer inherit from this class.
    """
    NUM_WORKERS = 4

    DEFAULT_CKPT_SUFFIX = '-ckpt.pt'
    BEST_VAL_ACC_CKPT_SUFFIX = '-best-val-acc-ckpt.pt'

    def __init__(self,
                 net_name,
                 model,
                 trainset,
                 save_path,
                 criterion,
                 loss_func,
                 optimizers,
                 mode='classify',
                 save_acc_by_set_size_by_epoch=False,
                 batch_size=64,
                 epochs=200,
                 use_val=False,
                 valset=None,
                 val_step=None,
                 patience=20,
                 ckpt_step=10,
                 summary_step=None,
                 sigmoid_threshold=0.5,
                 device='cuda',
                 num_workers=NUM_WORKERS,
                 data_parallel=False,
                 ):
        """returns new trainer instance

        Parameters
        ----------
        net_name : str
            name of neural network architecture. Used when saving model, checkpoints, etc.
        model : torch.nn.Module
            actual instance of network.
        trainset : torch.Dataset or torchvision.Visiondataset
            training data, represented as a class.
        save_path : Path
            path to directory where trained models and checkpoints (if any) should be saved
        save_acc_by_set_size_by_epoch : bool
            if True, compute and save accuracy for each visual search set size for each epoch
        criterion : torch.nn._Loss subclass
        loss_func : str
            that represents loss function and target that should be used with it.
            Used to determine targets for computing loss, and for metrics to use
            when determining whether to stop early due to metrics computed on
            validation set.
        optimizers : list
            of optimizers, e.g. torch.nn.SGD
        batch_size : int
            number of training samples per batch
        epochs : int
            number of epochs to train network
        val_step : int
            epoch at which accuracy should be measured on validation step.
            Validation occurs every time epoch % val_step == 0.
        use_val : bool
            if True, use validation set. Default is False.
        valset : torch.Dataset or torchvision.VisionDataset
            validation data, represented as a class.
        patience : int
            number of validation epochs to wait before stopping training if accuracy does not increase
        ckpt_step : int
            epoch at which to save a checkpoint.
            Occurs every time epoch % ckpt_step == 0.
        summary_step : int
            step at which to save summary to file.
            Occurs every time step % summary_step == 0.
            Each minibatch is considered a step, and steps are counted across epochs.
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
        self.net_name = net_name  # for checkpointing, saving model

        self.data_parallel = data_parallel
        if data_parallel:
            model = nn.DataParallel(model)
        model.to(device)
        self.model = model
        self.device = device
        self.trainset = trainset
        self.train_loader = DataLoader(self.trainset, batch_size=batch_size,
                                       shuffle=True, num_workers=num_workers,
                                       pin_memory=True)

        if use_val:
            self.valset = valset
            self.val_loader = DataLoader(self.valset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
        else:
            self.valset = None
            self.val_loader = None

        self.batch_size = batch_size
        self.step = 0  # each minibatch is a step, and we count steps across epochs
        self.epochs = epochs
        self.val_step = val_step
        self.patience = patience
        if self.patience is not None:
            self.best_val_acc = 0
            self.steps_without_improvement = 0
        self.ckpt_step = ckpt_step

        self.mode = mode

        self.save_acc_by_set_size_by_epoch = save_acc_by_set_size_by_epoch
        if save_acc_by_set_size_by_epoch:
            self.set_sizes = np.unique(self.trainset.set_size)
            self.acc_epoch_set_size_savepath = str(save_path) + '_acc_by_epoch_by_set_size.txt'
            self.acc_by_epoch_by_set_size = np.full(shape=(epochs, self.set_sizes.shape[0]), fill_value=np.nan)

        criterion.to(device)
        self.criterion = criterion
        self.loss_func = loss_func
        self.optimizers = optimizers
        self.save_path = save_path
        self.summary_step = summary_step
        if summary_step:
            self.train_writer = SummaryWriter(
                log_dir=str(Path(self.save_path).joinpath('train'))
            )
        else:
            self.train_writer = None

        self.sigmoid_threshold = sigmoid_threshold
        if isinstance(self.trainset, datasets.VOCDetection) or mode == 'detect':
            self.sigmoid_activation = torch.nn.Sigmoid()
        else:
            self.sigmoid_activation = None

    def save_checkpoint(self, epoch, ckpt_path=None):
        print(f'Saving checkpoint in {self.save_path}')
        ckpt = {
            'epoch': epoch,
            'step': self.step,
            'model': self.model.state_dict(),
        }
        for ind, optimizer in enumerate(self.optimizers):
            ckpt[f'optimizer_{ind}'] = optimizer.state_dict()
        if ckpt_path is None:
            ckpt_path = self.save_path.parent.joinpath(self.save_path.name + self.DEFAULT_CKPT_SUFFIX)
        torch.save(ckpt, str(ckpt_path))  # torch.save expects path as a string

    def train(self):
        for epoch in range(1, self.epochs + 1):

            print(f'\nEpoch {epoch}')
            self.train_one_epoch(epoch=epoch)
            if self.patience is not None:
                if self.steps_without_improvement > self.patience:
                    # need to break here, in addition to inside train_one_epoch method
                    break

            if self.save_acc_by_set_size_by_epoch:
                # --- compute accuracy on whole training set, by set size, for this epoch
                print('Computing accuracy per visual search stimulus set size on training set')
                self.train_acc_by_set_size(epoch)

        # --------------- done training, save checkpoint again + training history info -------------------------------
        self.save_checkpoint(epoch)

        if self.save_acc_by_set_size_by_epoch:
            # and save matrix with accuracy by epoch by set size
            np.savetxt(self.acc_epoch_set_size_savepath,
                       self.acc_by_epoch_by_set_size,
                       delimiter=',')

    def train_one_epoch(self, epoch):
        """train model for one epoch"""
        self.model.train()

        total_loss = 0.0

        batch_total = int(np.ceil(len(self.trainset) / self.batch_size))
        batch_pbar = tqdm(self.train_loader)

        for i, batch in enumerate(batch_pbar):
            self.step += 1

            if self.mode == 'classify':
                batch_x = batch['img'].to(self.device)
                if self.loss_func == 'BCE' or self.loss_func == 'CE':
                    batch_y = batch['target'].to(self.device)
                elif self.loss_func == 'CE-largest':
                    batch_y = batch['largest'].to(self.device)
                elif self.loss_func == 'CE-random':
                    batch_y = batch['random'].to(self.device)

                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)

            elif self.mode == 'detect':
                img, target_one_hot = batch['img'].to(self.device), batch['target']
                batch_size, n_classes = target_one_hot.shape
                # compute for every batch since last batch can be smaller
                half_batch_size = int(batch_size / 2)

                # -- make half of batch be target present, half target absent --
                permuted_sample_inds = np.random.permutation(batch_size)
                target_present_inds = permuted_sample_inds[:half_batch_size]
                target_absent_inds = permuted_sample_inds[half_batch_size:]
                target = np.zeros((batch_size, 1)).astype(np.float32)
                target[target_present_inds] = 1
                target[target_absent_inds] = 0

                query = np.zeros((batch_size, n_classes)).astype(np.float32)
                query_one_hot = torch.diag(torch.ones(n_classes,))
                for row, (target_present, classes_present_one_hot) in enumerate(zip(target, target_one_hot)):
                    if target_present == 0:
                        candidate_inds = np.nonzero(classes_present_one_hot == 0).flatten()
                    elif target_present == 1:
                        candidate_inds = np.nonzero(classes_present_one_hot == 1).flatten()
                    choice = np.random.choice(candidate_inds)
                    query[row] = query_one_hot[choice, :]

                query = torch.from_numpy(query).to(self.device)
                target = torch.from_numpy(target).to(self.device)

                output = self.model(img, query)
                loss = self.criterion(output, target)

            for optimizer in self.optimizers:
                optimizer.zero_grad()
            loss.mean().backward()  # mean needed for multiple GPUs
            for optimizer in self.optimizers:
                optimizer.step()

            batch_pbar.set_description(f'batch {i} of {batch_total}, loss: {loss: 7.3f}')
            total_loss += loss

            if self.summary_step:
                if self.step % self.summary_step == 0:
                    self.train_writer.add_scalar('loss/train', loss.mean(), self.step)

            if self.val_loader is not None:
                if self.step % self.val_step == 0:

                    val_metrics = self.validate()
                    self.model.train()  # switch back to train after validate calls eval
                    if self.mode == 'classify':
                        if self.loss_func == 'CE':
                            val_acc_this_epoch = val_metrics['acc']
                        elif self.loss_func == 'BCE':
                            val_acc_this_epoch = val_metrics['f1']
                        elif self.loss_func == 'CE-largest':
                            val_acc_this_epoch = val_metrics['acc_largest']
                        elif self.loss_func == 'CE-random':
                            val_acc_this_epoch = val_metrics['acc_random']
                    elif self.mode == 'detect':
                        val_acc_this_epoch = val_metrics['acc']

                    if self.patience is not None:
                        if val_acc_this_epoch > self.best_val_acc:
                            self.best_val_acc = val_acc_this_epoch
                            self.steps_without_improvement = 0
                            print(f'Validation accuracy improved, saving model in {self.save_path}')
                            # notice here we use a different "suffix" so there will be *two* checkpoint files
                            # one of which is the best validation accuracy, and the other saved on the checkpoint step
                            # (if specified) and at the end of training
                            self.save_checkpoint(
                                epoch=epoch,
                                ckpt_path=self.save_path.parent.joinpath(
                                    self.save_path.name + self.BEST_VAL_ACC_CKPT_SUFFIX
                                )
                            )
                        else:
                            self.steps_without_improvement += 1
                            if self.steps_without_improvement > self.patience:
                                print(
                                    f'greater than {self.patience} steps without improvement '
                                    'in validation accuracy, stopping training')
                                break

            if self.ckpt_step:
                if self.step % self.ckpt_step == 0:
                    self.save_checkpoint(epoch)

        avg_loss = total_loss / batch_total
        print(f'\tTraining Avg. Loss: {avg_loss:7.3f}')

    def validate(self):
        self.model.eval()

        val_loss = []

        if self.mode == 'classify':
            if isinstance(self.val_loader.dataset, datasets.Searchstims):
                val_acc = []
            elif isinstance(self.val_loader.dataset, datasets.VOCDetection):
                val_f1 = []
                val_acc_largest = []
                val_acc_random = []
        elif self.mode == 'detect':
            val_acc = []
            half_batch_size = int(self.batch_size / 2)

        with torch.no_grad():
            total = int(np.ceil(len(self.valset) / self.batch_size))
            pbar = tqdm(self.val_loader)
            for i, batch in enumerate(pbar):
                pbar.set_description(f'batch {i} of {total}')

                # ---- get output, compute loss ----
                if self.mode == 'classify':
                    batch_x = batch['img'].to(self.device)
                    if self.loss_func == 'BCE' or self.loss_func == 'CE':
                        batch_y = batch['target'].to(self.device)
                    elif self.loss_func == 'CE-largest':
                        batch_y = batch['largest'].to(self.device)
                    elif self.loss_func == 'CE-random':
                        batch_y = batch['random'].to(self.device)

                    output = self.model(batch_x)
                    loss = self.criterion(output, batch_y)

                elif self.mode == 'detect':
                    img, target_one_hot = batch['img'].to(self.device), batch['target']
                    batch_size, n_classes = target_one_hot.shape
                    # compute for every batch since last batch can be smaller
                    half_batch_size = int(batch_size / 2)

                    # -- make half of batch be target present, half target absent --
                    permuted_sample_inds = np.random.permutation(batch_size)
                    target_present_inds = permuted_sample_inds[:half_batch_size]
                    target_absent_inds = permuted_sample_inds[half_batch_size:]
                    target = np.zeros((batch_size, 1)).astype(np.float32)
                    target[target_present_inds] = 1
                    target[target_absent_inds] = 0

                    query = np.zeros((batch_size, n_classes)).astype(np.float32)
                    query_one_hot = torch.diag(torch.ones(n_classes, ))
                    for row, (target_present, classes_present_one_hot) in enumerate(zip(target, target_one_hot)):
                        if target_present == 0:
                            candidate_inds = np.nonzero(classes_present_one_hot == 0).flatten()
                        elif target_present == 1:
                            candidate_inds = np.nonzero(classes_present_one_hot == 1).flatten()
                        choice = np.random.choice(candidate_inds)
                        query[row] = query_one_hot[choice, :]

                    query = torch.from_numpy(query).to(self.device)
                    target = torch.from_numpy(target).to(self.device)

                    output = self.model(img, query)
                    loss = self.criterion(output, target)

                val_loss.append(loss.mean().cpu())  # mean needed for multiple GPUs

                # ---- compute accuracy / accuracies ----
                if self.mode == 'classify':
                    # below, _ because torch.max returns (values, indices)
                    _, pred_max = torch.max(output.data, 1)
                    if isinstance(self.val_loader.dataset, datasets.Searchstims):
                        acc = (pred_max == batch_y).sum().item() / batch_y.size(0)
                        val_acc.append(acc)
                    elif isinstance(self.val_loader.dataset, datasets.VOCDetection):
                        batch_y_onehot = batch['target'].to(self.device)
                        out_sig = self.sigmoid_activation(output)
                        pred_sig = (out_sig > self.sigmoid_threshold).float()
                        f1 = sklearn.metrics.f1_score(batch_y_onehot.cpu().numpy(), pred_sig.cpu().numpy(),
                                                      average='macro')
                        val_f1.append(f1)

                        batch_largest = batch['largest'].to(self.device)
                        acc_largest = (pred_max == batch_largest).sum().item() / batch_largest.size(0)
                        val_acc_largest.append(acc_largest)

                        batch_random = batch['random'].to(self.device)
                        acc_random = (pred_max == batch_random).sum().item() / batch_random.size(0)
                        val_acc_random.append(acc_random)

                elif self.mode == 'detect':
                    out_sig = self.sigmoid_activation(output)
                    pred_sig = (out_sig > self.sigmoid_threshold).float()
                    acc = (pred_sig == target).sum().item() / target.size(0)
                    val_acc.append(acc)

        # ---- assemble dict of metrics to return ----
        val_metrics = {
            'loss': np.asarray(val_loss).mean(),
        }
        if self.mode == 'classify':
            if isinstance(self.val_loader.dataset, datasets.Searchstims):
                val_metrics['acc'] = np.asarray(val_acc).mean()
            elif isinstance(self.val_loader.dataset, datasets.VOCDetection):
                val_metrics['f1'] = np.asarray(val_f1).mean()
                val_metrics['acc_largest'] = np.asarray(val_acc_largest).mean()
                val_metrics['acc_random'] = np.asarray(val_acc_random).mean()
        elif self.mode == 'detect':
            val_metrics['acc'] = np.asarray(val_acc).mean()

        metrics_str = ', '.join(
            [f'{metric}:{value:7.3f}' for metric, value in val_metrics.items()]
        )

        print(
            f' Validation: {metrics_str}'
        )

        if self.summary_step:
            for metric, value in val_metrics.items():
                self.train_writer.add_scalar(f'{metric}/val', value, self.step)

        return val_metrics

    def train_acc_by_set_size(self, epoch):
        self.model.eval()

        set_sizes = np.unique(self.trainset.set_size)
        acc_by_set_size = {set_size: [] for set_size in set_sizes}

        batch_total = int(np.ceil(len(self.trainset) / self.batch_size))
        pbar = tqdm(self.train_loader)
        for i, batch in enumerate(pbar):
            pbar.set_description(f'batch {i} of {batch_total}')
            batch_x = batch['img'].to(self.device)
            batch_y = batch['target'].to(self.device)
            batch_set_size = batch['set_size'].cpu().numpy()

            output = self.model(batch_x)
            _, predictions = torch.max(output, 1)
            correct = (predictions == batch_y).cpu().numpy()
            for set_size in np.unique(batch_set_size):
                inds = np.nonzero(batch_set_size == set_size)[0]
                acc = correct[inds].sum() / inds.shape[0]
                acc_by_set_size[set_size].append(acc)

        acc_by_set_size = {set_size: np.asarray(accs).mean()
                           for set_size, accs in acc_by_set_size.items()}

        acc_set_size_str = ''.join(
            [f'set size {set_size}: {acc}. ' for set_size, acc in acc_by_set_size.items()]
        )
        print(acc_set_size_str)

        for set_size, acc in acc_by_set_size.items():
            row = epoch - 1  # because we start counting epochs at 1
            col = np.nonzero(self.set_sizes == set_size)[0]
            self.acc_by_epoch_by_set_size[row, col] = acc
