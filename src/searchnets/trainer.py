"""Trainer class"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from .utils.dataset import VisSearchDataset
from . import nets
# from .triplet_loss import batch_all_triplet_loss, dist_squared, dist_euclid

MOMENTUM = 0.9  # used for both Alexnet and VGG16
NUM_WORKERS = 4

# for preprocessing, normalize using values used when training these models on ImageNet for torchvision
# see https://github.com/pytorch/examples/blob/632d385444ae16afe3e4003c94864f9f97dc8541/imagenet/main.py#L197-L198
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Trainer:
    """class for training CNNs on visual search task"""
    def __init__(self,
                 net_name,
                 new_learn_rate_layers,
                 csv_file,
                 save_path,
                 loss_func='ce',
                 save_acc_by_set_size_by_epoch=True,
                 freeze_trained_weights=False,
                 base_learning_rate=1e-20,
                 new_layer_learning_rate=0.00001,
                 batch_size=64,
                 epochs=200,
                 val_epoch=1,
                 use_val=True,
                 patience=20,
                 checkpoint_epoch=10,
                 summary_step=None,
                 device='cuda',
                 num_workers=NUM_WORKERS,
                 ):
        """

        Parameters
        ----------
        net_name
        new_learn_rate_layers
        csv_file
        save_path
        loss_func
        save_acc_by_set_size_by_epoch
        freeze_trained_weights
        base_learning_rate
        new_layer_learning_rate
        batch_size
        epochs
        val_epoch
        use_val
        patience
        checkpoint_epoch
        summary_step
        device
        num_workers
        """
        self.net_name = net_name  # for checkpointing, saving model
        if net_name == 'alexnet':
            model = nets.alexnet.build(pretrained=True, progress=True)
            model = nets.alexnet.reinit(model, new_learn_rate_layers)
        elif net_name == 'VGG16':
            model = nets.vgg16.build(pretrained=True, progress=True)
            model = nets.vgg16.reinit(model, new_learn_rate_layers)

        model.to(device)
        self.model = model
        self.device = device

        normalize = transforms.Normalize(mean=MEAN,
                                         std=STD)

        self.trainset = VisSearchDataset(csv_file=csv_file,
                                         split='train',
                                         transform=transforms.Compose(
                                             [transforms.ToTensor(), normalize]
                                         ))
        self.train_loader = DataLoader(self.trainset, batch_size=batch_size,
                                       shuffle=True, num_workers=num_workers,
                                       pin_memory=True)

        self.csv_file = csv_file
        self.dataset_df = pd.read_csv(csv_file)

        if use_val:
            self.valset = VisSearchDataset(csv_file=csv_file,
                                           split='val',
                                           transform=transforms.Compose([transforms.ToTensor(), normalize]))
            self.val_loader = DataLoader(self.valset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
        else:
            self.val_loader = None

        self.batch_size = batch_size
        self.step = 0  # each minibatch is a step, and we count steps across epochs
        self.epochs = epochs
        self.val_epoch = val_epoch
        self.patience = patience
        self.checkpoint_epoch = checkpoint_epoch

        self.save_acc_by_set_size_by_epoch = save_acc_by_set_size_by_epoch
        if save_acc_by_set_size_by_epoch:
            self.set_sizes = self.dataset_df['set_size'].unique()
            self.acc_epoch_set_size_savepath = str(save_path) + '_acc_by_epoch_by_set_size.txt'
            self.trainset_set_size = VisSearchDataset(csv_file=csv_file,
                                                      split='train',
                                                      transform=transforms.Compose(
                                                          [transforms.ToTensor(), normalize]),
                                                      return_set_size=True)
            self.train_loader_no_shuffle = DataLoader(self.trainset_set_size, batch_size=batch_size,
                                                      shuffle=False, num_workers=num_workers)
            self.acc_by_epoch_by_set_size = np.full(shape=(epochs, self.set_sizes.shape[0]), fill_value=np.nan)

        else:
            self.train_loader_no_shuffle = None

        if loss_func == 'CE':
            self.criterion = nn.CrossEntropyLoss().to(device)
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

        self.optimizers = []
        classifier_params = model.classifier.parameters()
        if freeze_trained_weights:
            self.optimizers.append(
                torch.optim.SGD(classifier_params,
                                lr=new_layer_learning_rate,
                                momentum=MOMENTUM))
            for params in model.features.parameters():
                params.requires_grad = False
        else:
            self.optimizers.append(
                torch.optim.SGD(classifier_params,
                                lr=new_layer_learning_rate,
                                momentum=MOMENTUM)
            )
            feature_params = model.features.parameters()
            self.optimizers.append(
                torch.optim.SGD(feature_params,
                                lr=base_learning_rate,
                                momentum=MOMENTUM)
            )

        self.save_path = save_path

        self.summary_step = summary_step
        if summary_step:
            self.train_writer = SummaryWriter(
                log_dir=str(Path(self.save_path).joinpath('train'))
            )
        else:
            self.train_writer = None

    def save_checkpoint(self, epoch):
        print(f'Saving checkpoint in {self.save_path}')
        ckpt = {
            'epoch': epoch,
            'step': self.step,
            'model': self.model.state_dict(),
        }
        for ind, optimizer in enumerate(self.optimizers):
            ckpt[f'optimizer_{ind}'] = optimizer.state_dict()
        ckpt_file = str(self.save_path) + '-ckpt.tar'
        torch.save(ckpt, ckpt_file)

    def save_model(self):
        model_file = str(self.save_path) + '-model.pt'
        torch.save(self.model.state_dict(), model_file)

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
                            self.save_model()
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
                if epoch % self.checkpoint_epoch == 0:
                    self.save_checkpoint(epoch)

            if self.save_acc_by_set_size_by_epoch:
                # --- compute accuracy on whole training set, by set size, for this epoch
                print('Computing accuracy per visual search stimulus set size on training set')
                self.train_acc_by_set_size(epoch)

        # --------------- done training, save model + training history info -------------------------------
        if self.patience is None:
            # only save at end if we haven't already been saving checkpoints
            print(f'Saving model in {self.save_path}')
            self.save_model()

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
            loss.backward()
            for optimizer in self.optimizers:
                optimizer.step()

            batch_pbar.set_description(f'batch {i} of {batch_total}, loss: {loss: 7.3f}')
            total_loss += loss

            if self.summary_step:
                if self.step % self.summary_step == 0:
                    print('would save summary here')
                    # train_summaries = []
                    # embeddings = model.fc7
                    # # t = target, d = distractor
                    # t_inds = tf.where(tf.math.equal(y, 1))
                    # t_vecs = tf.gather(embeddings, t_inds)
                    # t_vecs = tf.squeeze(t_vecs)
                    # if squared_dist:
                    #     t_distances = dist_squared(t_vecs)
                    # else:
                    #     t_distances = dist_euclid(t_vecs)
                    # train_summaries.extend([
                    #     tf.summary.histogram('target_distances', t_distances),
                    #     tf.summary.scalar('target_distances_mean', tf.reduce_mean(t_distances)),
                    #     tf.summary.scalar('target_distances_std', tf.math.reduce_std(t_distances)),
                    # ])
                    #
                    # d_inds = tf.where(tf.math.equal(y, 0))
                    # d_vecs = tf.gather(embeddings, d_inds)
                    # d_vecs = tf.squeeze(d_vecs)
                    # if squared_dist:
                    #     d_distances = dist_squared(d_vecs)
                    # else:
                    #     d_distances = dist_euclid(d_vecs)
                    # train_summaries.extend([
                    #     tf.summary.histogram('distractor_distances', d_distances),
                    #     tf.summary.scalar('distractor_distances_mean', tf.reduce_mean(d_distances)),
                    #     tf.summary.scalar('distractor_distances_std', tf.math.reduce_std(d_distances)),
                    # ])
                    #
                    # if squared_dist:
                    #     td_distances = dist_squared(t_vecs, d_vecs)
                    # else:
                    #     td_distances = dist_euclid(t_vecs, d_vecs)
                    # train_summaries.extend([
                    #     tf.summary.histogram('target_distractor_distances', td_distances),
                    #     tf.summary.scalar('target_distractor_distances_mean', tf.reduce_mean(td_distances)),
                    #     tf.summary.scalar('target_distractor_distances_std', tf.math.reduce_std(td_distances)),
                    # ])
                    # self.train_writer.add_summary(summary, step)

        avg_loss = total_loss / batch_total
        print(f'\tTraining Avg. Loss: {avg_loss:7.3f}')

    def validate(self):
        self.model.eval()

        val_acc_this_epoch = []
        with torch.no_grad():
            total = int(np.ceil(len(self.valset) / self.batch_size))
            pbar = tqdm(self.val_loader)
            for i, (batch_x, batch_y) in enumerate(pbar):
                pbar.set_description(f'batch {i} of {total}')
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                # below, _ because torch.max returns (values, indices)
                _, predicted = torch.max(output.data, 1)
                acc = (predicted == batch_y).sum().item() / batch_y.size(0)
                val_acc_this_epoch.append(acc)

        val_acc_this_epoch = np.asarray(val_acc_this_epoch).mean()
        print(' Validation Acc: %7.3f' % val_acc_this_epoch)

        if self.summary_step:
            print('would write val summary')
            # self.train_writer.add_summary(val_acc_summary, step)

        return val_acc_this_epoch

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
