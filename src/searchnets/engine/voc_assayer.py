from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import nets
from ..analysis.searchstims import compute_d_prime
from ..engine.abstract_trainer import AbstractTrainer
from ..transforms.functional import tile


class VOCAssayer:
    """class for running "behavioral assay" of models using Pascal VOC / Visual Search Difficulty dataset"""
    NUM_WORKERS = 4

    def __init__(self,
                 net_name,
                 model,
                 loss_func,
                 testset,
                 restore_path,
                 mode='classify',
                 batch_size=64,
                 sigmoid_threshold=0.5,
                 device='cuda',
                 num_workers=NUM_WORKERS,
                 data_parallel=False,
                 ):
        """

        Parameters
        ----------
        net_name : str
            name of convolutional neural net architecture to train.
            One of {'alexnet', 'VGG16'}
        model : torch.nn.Module
            actual instance of network.
        loss_func : str
            that represents loss function and target that should be used with it.
            Used to determine targets for computing loss, and for metrics to use
            when determining whether to stop early due to metrics computed on
            validation set.
        testset : torch.Dataset or torchvision.Visiondataset
            test data, represented as a class.
        restore_path : Path
            path to directory where checkpoints and train models were saved
        mode : str
            training mode. One of {'classify', 'detect'}.
            'classify' is standard image classification.
            'detect' trains to detect whether specified target is present or absent.
            Default is 'classify'.
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

        self.mode = mode
        self.batch_size = batch_size
        self.sigmoid_threshold = sigmoid_threshold
        self.sigmoid_activation = torch.nn.Sigmoid()
        self.softmax_activation = torch.nn.Softmax()

        self.loss_func = loss_func

    @classmethod
    def from_config(cls,
                    net_name,
                    num_classes,
                    loss_func,
                    testset,
                    mode='classify',
                    embedding_n_out=512,
                    **kwargs):
        """factory function that creates instance of VOCAssayer from options specified in config.ini file

        Parameters
        ----------
        net_name : str
            name of neural network architecture. Used when restoring model, checkpoints, etc.
        num_classes : int
            number of classes. Default is 2 (target present, target absent).
        loss_func : str
            type of loss function to use. One of {'CE', 'InvDPrime', 'triplet'}. Default is 'CE',
            the standard cross-entropy loss. 'InvDPrime' is inverse D prime. 'triplet' is triplet loss
            used in face recognition and biometric applications.
        testset : torch.Dataset or torchvision.Visiondataset
            split of dataset for testing model, represented as a class.
        mode : str
            training mode. One of {'classify', 'detect'}.
            'classify' is standard image classification.
            'detect' trains to detect whether specified target is present or absent.
            Default is 'classify'.
        embedding_n_out : int
            for DetectNet, number of output features from input embedding.
            I.e., the output size of the linear layer that accepts the
            one hot vector querying whether a specific class is present as input.
            Default is 512.
        kwargs : keyword arguments to VOCAssayer

        Returns
        -------
        assayer : VOCAssayer
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

        if mode == 'detect':
            # remove final output layer, will replace
            if net_name == 'alexnet' or net_name == 'VGG16':
                model.classifier = model.classifier[:-1]
            elif 'cornet' in net_name.lower():
                # for CORnet models, also need to remove 'output' layer (just an Identity)
                model.decoder = model.decoder[:-2]
            a_sample = next(iter(testset))
            tmp_img = a_sample['img'].unsqueeze(0)  # add batch dim
            tmp_out = model(tmp_img)
            vis_sys_n_features_out = tmp_out.shape[-1]  # (batch, n features)
            model = nets.detectnet.DetectNet(vis_sys=model,
                                             num_classes=num_classes,
                                             vis_sys_n_out=vis_sys_n_features_out,
                                             embedding_n_out=embedding_n_out)

        if loss_func in {'CE', 'CE-largest', 'CE-random'}:
            criterion = nn.CrossEntropyLoss()
        elif loss_func == 'BCE':
            criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(
                f'invalid value for loss function: {loss_func}'
            )

        return cls(net_name=net_name,
                   model=model,
                   loss_func=loss_func,
                   mode=mode,
                   testset=testset,
                   **kwargs)

    def assay(self):
        """assay behavior of trained model

        Returns
        -------
        results : dict
            with following key-value pairs:
                arrays : dict
                    inputs / outputs to networks as numpy arrays
                        y_true_onehot : numpy.ndarray
                            true classes present in image, encoded as "one" hot vectors
                            (actually can be more than one class present)
                        out : numpy.ndarray
                            output of network after non-linear activation has been applied,
                            sigmoid if trained with binary cross entropy loss,
                            or softmax if trained with cross entropy
                        y_pred : numpy.ndarray
                            prediction after thresholding (for sigmoid)
                            or finding argmax (softmax) of output
                df_test : pandas.DataFrame
                    where each row is a sample from the dataset
        """
        self.model.eval()

        total = int(np.ceil(len(self.testset) / self.batch_size))
        pbar = tqdm(self.test_loader)

        # lists of numpy arrays that get concatenated at the end,
        # save for further analysis if required
        arrays = defaultdict(list)

        # will use with Pandas.DataFrame.from_records()
        # to make dataframe of test results, where each row is one sample from test set
        image_records = defaultdict(list)
        trial_records = defaultdict(list)


        with torch.no_grad():
            for i, batch in enumerate(pbar):
                pbar.set_description(f'batch {i} of {total}')

                if self.mode == 'classify':
                    # ---- get outputs ----
                    x_batch, y_true_onehot_batch = batch['img'].to(self.device), batch['target'].to(self.device)
                    batch_size, n_classes = y_true_onehot_batch.shape  # used for np.split below
                    arrays['y_true_onehot'].append(y_true_onehot_batch.cpu().numpy())
                    out_batch = self.model(x_batch)

                    # ---- pass outputs through activation ----
                    if self.loss_func == 'BCE':
                        out_batch = self.sigmoid_activation(out_batch)
                        y_pred_batch = (out_batch > self.sigmoid_threshold).float()
                    elif self.loss_func == 'CE-largest' or self.loss_func == 'CE-random':
                        out_batch = self.softmax_activation(out_batch)
                        _, y_pred_batch = torch.max(out_batch.data, 1)
                        # -- convert to a one-hot representation to compute TP, FP, d prime, accuracy, etc. --
                        # make tensor below for every batch, in case it changes size (e.g. for last batch)
                        y_pred_softmax_onehot_batch = torch.FloatTensor(y_true_onehot_batch.shape).to(
                            y_true_onehot_batch.device  # just copy this tensor so it's the same shape / dtype
                        )
                        y_pred_softmax_onehot_batch.zero_()  # but then zero out
                        y_pred_softmax_onehot_batch.scatter_(1, torch.unsqueeze(y_pred_batch, 1), 1)
                        # note we make 'y_pred_batch' name point at the tensor we just made
                        y_pred_batch = y_pred_softmax_onehot_batch

                    # ---- save outputs to concatenate and return with results
                    arrays['out'].append(out_batch.cpu().numpy())  # raw output of network
                    arrays['y_pred'].append(y_pred_batch.cpu().numpy())

                    # ---- compute true positive, false positive, true negative, false negative
                    TP = ((y_true_onehot_batch == 1) & (y_pred_batch == 1)).sum(dim=1).cpu().numpy()
                    FP = ((y_true_onehot_batch == 0) & (y_pred_batch == 1)).sum(dim=1).cpu().numpy()
                    TN = ((y_true_onehot_batch == 0) & (y_pred_batch == 0)).sum(dim=1).cpu().numpy()
                    FN = ((y_true_onehot_batch == 1) & (y_pred_batch == 0)).sum(dim=1).cpu().numpy()

                elif self.mode == 'detect':
                    img_batch, target_batch = batch['img'], batch['target']
                    batch_size, n_classes = target_batch.shape  # used for this code block and np.split below
                    img_batch = tile(img_batch, dim=0, n_tile=n_classes)  # repeat each img n_classes number of times
                    # make a diagonal so we can query for every possible class in the image
                    query_expanded = torch.cat(batch_size * [torch.diag(torch.ones(n_classes, ))])
                    target_batch = target_batch.flatten()
                    target_batch = target_batch.unsqueeze(1)  # add back non-batch ind, so target matches output shape

                    img_batch = img_batch.to(self.device)
                    query_batch = query_expanded.to(self.device)
                    target_batch = target_batch.to(self.device)

                    # --- split into batches, keeping same batch size in case for same reason there's batch size fx
                    # for "assay", need to re-assemble outputs with same shape as we would get for 'classify'
                    out_batch = []
                    y_pred_batch = []
                    y_true_onehot_batch = []
                    for img, query, target in zip(
                            torch.split(img_batch, self.batch_size),
                            torch.split(query_batch, self.batch_size),
                            torch.split(target_batch, self.batch_size)
                    ):
                        y_true_onehot_batch.append(target.cpu().numpy())

                        out = self.model(img, query)
                        out = self.sigmoid_activation(out)
                        out_batch.append(out.cpu().numpy())

                        y_pred = (out > self.sigmoid_threshold).float()
                        y_pred_batch.append(y_pred.cpu().numpy())

                    out_batch = np.concatenate(out_batch).reshape(-1, n_classes)
                    y_pred_batch = np.concatenate(y_pred_batch).reshape(-1, n_classes)
                    y_true_onehot_batch = np.concatenate(y_true_onehot_batch).reshape(-1, n_classes)
                    for key, val in zip(
                            ('out', 'y_pred', 'y_true_onehot'),
                            (out_batch, y_pred_batch, y_true_onehot_batch)
                    ):
                        arrays[key].append(val)

                    TP = ((y_true_onehot_batch == 1) & (y_pred_batch == 1)).sum(axis=1)
                    FP = ((y_true_onehot_batch == 0) & (y_pred_batch == 1)).sum(axis=1)
                    TN = ((y_true_onehot_batch == 0) & (y_pred_batch == 0)).sum(axis=1)
                    FN = ((y_true_onehot_batch == 1) & (y_pred_batch == 0)).sum(axis=1)

                # now loop through each sample in batch to add to records; these will be rows in dataframe
                index_batch = batch['index']
                img_indices_list = index_batch.cpu().numpy().tolist()
                img_paths = [
                    Path(self.testset.images[idx]) for idx in img_indices_list
                ]
                img_names = [img_path.name for img_path in img_paths]
                n_items = arrays['y_true_onehot'][-1].sum(axis=1).astype(int)

                vsd_score_batch = batch['vsd_score']
                zip_for_loop = zip(
                    torch.unbind(index_batch),
                    img_paths,
                    img_names,
                    torch.unbind(vsd_score_batch),
                    TP.tolist(),
                    FP.tolist(),
                    TN.tolist(),
                    FN.tolist(),
                    n_items.tolist(),
                    # below, by zipping these arrays, we get one row for each step in iteration
                    arrays['out'][-1],
                    arrays['y_pred'][-1],
                    arrays['y_true_onehot'][-1],
                )
                for tuple in zip_for_loop:
                    image_row = dict(zip(
                        ['voc_test_index', 'img_path', 'img_name', 'vsd_score',
                         'TP', 'FP', 'TN', 'FN', 'n_items'],
                        tuple[:-3])
                    )
                    for key in image_row.keys():
                        value = image_row[key]
                        if isinstance(value, torch.Tensor):
                            value = value.cpu().numpy().item()
                            image_row[key] = value
                        image_records[key].append(value)
                    # now treat each class as a trial, regardless of mode, and add as a row to trial_records
                    probs, preds, target_present_vec = tuple[-3:]
                    for class_ind, (prob, pred, target_present) in enumerate(zip(
                            probs.tolist(), preds.tolist(), target_present_vec.tolist())):
                        for key, value in image_row.items():
                            trial_records[key].append(value)
                        trial_records['class'].append(class_ind)
                        trial_records['prob'].append(prob)
                        trial_records['pred'].append(pred)
                        trial_records['target_present'].append(target_present)

        arrays = {k: np.concatenate(v) for k, v in arrays.items()}
        images_df = pd.DataFrame.from_records(image_records)
        trials_df = pd.DataFrame.from_records(trial_records)

        y_pred_all = arrays['y_pred'].ravel()
        y_true_all = arrays['y_true_onehot'].ravel()
        acc = accuracy_score(y_pred=y_pred_all, y_true=y_true_all)
        _, _, d_prime = compute_d_prime(y_pred=y_pred_all, y_true=y_true_all)

        return {
            'arrays': arrays,
            'images_df': images_df,
            'trials_df': trials_df,
            'acc': acc,
            'd_prime': d_prime,
        }
