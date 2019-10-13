"""transforms used with Torch Datasets"""
import numpy as np
import torch
from torchvision import transforms

# for preprocessing, normalize using values used when training these models on ImageNet for torchvision
# see https://github.com/pytorch/examples/blob/632d385444ae16afe3e4003c94864f9f97dc8541/imagenet/main.py#L197-L198
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=MEAN,
                                 std=STD)

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCTransform:
    """Pads VOC image "randomly", so that size is constant but location of image is random,
    and transforms bounding box annotation for the image into a vector of classes


    Attributes
    ----------
    class_to_ind : dict
        dictionary lookup of classnames -> indexes
        (default: alphabetic indexing of VOC's 20 classes)
    """
    def __init__(self,
                 class_to_ind=None,
                 pad_size=500,
                 ):
        """
        Parameters
        ----------
        class_to_ind : dict
            dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        pad_size : int
            Size of image after padding. Default is 500, maximum size of VOC images.
        """
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.pad_size = pad_size
        self.to_tensor = transforms.ToTensor()

    def _random_pad(self, img):
        c, h, w = img.shape
        if h > self.pad_size:
            raise ValueError(
                f'height of image {h} is greater than pad size {self.pad_size}'
            )
        if w > self.pad_size:
            raise ValueError(
                f'width of image {w} is greater than pad size {self.pad_size}'
            )
        padded = torch.FloatTensor(c, self.pad_size, self.pad_size)
        padded.zero_()
        h_range = self.pad_size - h
        if h_range > 0:
            y_topleft = torch.randint(h_range, (1,))
        else:
            y_topleft = 0
        w_range = self.pad_size - w
        if w_range > 0:
            x_topleft = torch.randint(w_range, (1,))
        else:
            x_topleft = 0

        padded[:, y_topleft:y_topleft + h, x_topleft:x_topleft + w] = img

        return padded

    def __call__(self, img, target):
        """converts img to Tensor, performs a random crop, and normalizes using ImageNet mean and standard deviation.
        converts target to one-hot labels.

        Parameters
        ----------
        img : PIL.image
            image from VOC dataset
        target : dict
            target annotation, .xml annotation file loaded by Elementree and then converted to Python dict
            by VOCDetection. Will be converted into a Tensor of bounding box co-ordinates and integer label

        Returns
        -------
        img, target_out : torch.Tensor
            img, converted to Tensor, randomly padded, and then normalized.
            target, converted from .xml annotation to one-hot encoding of objects present after cropping.
        """
        img = self.to_tensor(img)
        img = self._random_pad(img)

        target_out = []
        objects = target['annotation']['object']
        if type(objects) == dict:  # if just a single object, not a list of them
            objects = [objects]  # wrap in a list so we can iterate over it
        for obj in objects:  # will be a list
            name = obj['name']
            label_idx = self.class_to_ind[name]
            target_out.append(label_idx)

        # we don't use functional.one_hot because we want to return an array of all zeros if none of the objects
        # are present; can't pass an array of "nothing" to functional.one_hot to get that output
        onehot = torch.FloatTensor(len(self.class_to_ind))  # number of classes
        onehot.zero_()
        target_out = onehot.scatter_(0, torch.LongTensor(target_out), 1)

        return img, target_out
