"""Trainer class"""
from pathlib import Path

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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
                 optimizers,
                 save_acc_by_set_size_by_epoch=False,
                 trainset_set_size=None,
                 batch_size=64,
                 epochs=200,
                 use_val=False,
                 valset=None,
                 val_epoch=None,
                 patience=20,
                 checkpoint_epoch=10,
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
        optimizers : list
            of optimizers, e.g. torch.nn.SGD
        batch_size : int
            number of training samples per batch
        epochs : int
            number of epochs to train network
        val_epoch : int
            epoch at which accuracy should be measured on validation step.
            Validation occurs every time epoch % val_epoch == 0.
        use_val : bool
            if True, use validation set. Default is False.
        valset : torch.Dataset or torchvision.VisionDataset
            validation data, represented as a class.
        patience : int
            number of validation epochs to wait before stopping training if accuracy does not increase
        checkpoint_epoch : int
            epoch at which to save a checkpoint.
            Occurs every time epoch % checkpoint_epoch == 0.
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
        self.val_epoch = val_epoch
        self.patience = patience
        self.checkpoint_epoch = checkpoint_epoch

        self.save_acc_by_set_size_by_epoch = save_acc_by_set_size_by_epoch
        if save_acc_by_set_size_by_epoch:
            self.trainset_set_size = trainset_set_size
            self.set_sizes = np.unique(self.trainset_set_size.set_size)
            self.acc_epoch_set_size_savepath = str(save_path) + '_acc_by_epoch_by_set_size.txt'
            self.train_loader_no_shuffle = DataLoader(self.trainset_set_size, batch_size=batch_size,
                                                      shuffle=False, num_workers=num_workers)
            self.acc_by_epoch_by_set_size = np.full(shape=(epochs, self.set_sizes.shape[0]), fill_value=np.nan)

        else:
            self.train_loader_no_shuffle = None

        criterion.to(device)
        self.criterion = criterion
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
        if self.val_loader is not None:
            val_acc = []
            if self.patience is not None:
                best_val_acc = 0
                epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):

            print(f'\nEpoch {epoch}')
            self.train_one_epoch()

            if self.val_loader is not None:
                if epoch % self.val_epoch == 0:

                    val_acc_this_epoch = self.validate()

                    if self.patience is not None:
                        if val_acc_this_epoch > best_val_acc:
                            best_val_acc = val_acc_this_epoch
                            epochs_without_improvement = 0
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
                            epochs_without_improvement += 1
                            if epochs_without_improvement > self.patience:
                                print(
                                    f'greater than {self.patience} epochs without improvement in validation '
                                    'accuracy, stopping training')

                                break
                else:  # if not a validation epoch
                    val_acc.append(None)

            if self.checkpoint_epoch:
                #
                if epoch % self.checkpoint_epoch == 0:
                    self.save_checkpoint(epoch)

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

    def train_one_epoch(self):
        """train model for one epoch"""
        self.model.train()

        total_loss = 0.0

        batch_total = int(np.ceil(len(self.trainset) / self.batch_size))
        batch_pbar = tqdm(self.train_loader)
        for i, (batch_x, batch_y) in enumerate(batch_pbar):
            self.step += 1

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)

            for optimizer in self.optimizers:
                optimizer.zero_grad()
            loss.mean().backward()  # mean needed for multiple GPUs
            for optimizer in self.optimizers:
                optimizer.step()

            batch_pbar.set_description(f'batch {i} of {batch_total}, loss: {loss: 7.3f}')
            total_loss += loss

            if self.summary_step:
                if self.step % self.summary_step == 0:
                    self.train_writer.add_scalar('Loss/train', loss.mean(), self.step)

        avg_loss = total_loss / batch_total
        print(f'\tTraining Avg. Loss: {avg_loss:7.3f}')

    def validate(self):
        self.model.eval()

        val_acc = []
        val_loss = []

        with torch.no_grad():
            total = int(np.ceil(len(self.valset) / self.batch_size))
            pbar = tqdm(self.val_loader)
            for i, (batch_x, batch_y) in enumerate(pbar):
                pbar.set_description(f'batch {i} of {total}')
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                val_loss.append(loss.mean().cpu())  # mean needed for multiple GPUs

                if batch_y.dim() > 1:
                    # convert to one hot vector
                    predicted = (output > self.sigmoid_threshold).float()
                    acc = sklearn.metrics.f1_score(batch_y.cpu().numpy(), predicted.cpu().numpy(),
                                                   average='macro')
                else:
                    # below, _ because torch.max returns (values, indices)
                    _, predicted = torch.max(output.data, 1)
                    acc = (predicted == batch_y).sum().item() / batch_y.size(0)
                val_acc.append(acc)

        val_acc = np.asarray(val_acc).mean()
        val_loss = np.asarray(val_loss).mean()
        print(
            f' Validation: accuracy {val_acc:7.3f} loss {val_loss:.4f}'
        )

        if self.summary_step:
            # just always add val acc regardless of step -- want to capture it every time
            self.train_writer.add_scalar('Acc/val', val_acc, self.step)
            self.train_writer.add_scalar('Loss/val', val_loss, self.step)

        return val_acc

    def train_acc_by_set_size(self, epoch):
        self.model.eval()

        set_sizes = np.unique(self.trainset_set_size.set_size)
        acc_by_set_size = {set_size: [] for set_size in set_sizes}

        batch_total = int(np.ceil(len(self.trainset_set_size) / self.batch_size))
        pbar = tqdm(self.train_loader_no_shuffle)
        for i, (batch_x, batch_y, batch_set_size) in enumerate(pbar):
            pbar.set_description(f'batch {i} of {batch_total}')
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
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
