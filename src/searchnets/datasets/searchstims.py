"""PyTorch Dataset class for visual search stimuli"""
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Searchstims(Dataset):
    """dataset of visual search stimuli"""

    def __init__(self,
                 csv_file,
                 split,
                 transform=None,
                 target_transform=None):
        """

        Parameters
        ----------
        csv_file : str
            name of .csv file generated by searchnets.data.split
        split : str
            Split of entire dataset to use. One of {'train', 'val', 'test'}.
        transform : callable
            transform to be applied to a single image from the dataset
        target_transform : callable
            transform to be applied to target
        """
        if split not in {'train', 'val', 'test'}:
            raise ValueError("split must be one of: {'train', 'val', 'test'}")

        self.csv_file = csv_file
        self.transform = transform
        self.split = split
        df = pd.read_csv(csv_file)
        df = df[df['split'] == split]
        self.df = df

        img_files = df['img_file'].values
        root_output_dir = df['root_output_dir'].values
        self.img_paths = np.asarray(
            [str(Path(root).joinpath(img_file))
             for root, img_file in zip(root_output_dir, img_files)]
        )

        target_condition = df['target_condition'].values
        target_condition = np.asarray(
            [1 if tc == 'present' else 0 for tc in target_condition]
        )
        self.target_condition = target_condition

        self.transform = transform
        self.target_transform = target_transform

        self.set_size = df['set_size'].values

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = imageio.imread(self.img_paths[idx])
        target = self.target_condition[idx]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        sample = {
            'img': img,
            'target': target,
            'set_size': self.set_size[idx],
        }

        return sample
