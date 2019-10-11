"""transforms used with Torch Datasets"""
from collections import namedtuple

import numpy as np
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

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


class VOCTransform:
    """Randomly crops VOC image, and transforms bounding box annotation for the image into a vector of classes
    that are present in the image after cropping

    Attributes
    ----------
    class_to_ind : dict
        dictionary lookup of classnames -> indexes
        (default: alphabetic indexing of VOC's 20 classes)
    threshold : float
        between 0 and 1. Amount of target bounding box that must still be within the image
        after cropping for it to be included in annotation.
        If None, overlap is not calculated and all annotations are included.
        Default is None.
    """
    def __init__(self,
                 class_to_ind=None,
                 random_crop=True,
                 crop_size=224,
                 threshold=None
                 ):
        """
        Parameters
        ----------
        class_to_ind : dict
            dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        threshold : float
            between 0 and 1. Amount of target bounding box that must still be within the image
            after cropping for it to be included in annotation.
            If None, overlap is not calculated and all annotations are included.
            Default is None.
        """
        if threshold is not None:
            if threshold < 0. or threshold > 1.:
                raise ValueError(
                    f'threshold must be between 0.0 and 1.0 but was {threshold}'
                )

        if self.random_crop and threshold is None:
            raise ValueError(
                'must specify threshold when random_crop is not None; otherwise, '
                'annotation may return labels for objects that are not present in image after it is cropped'
            )

        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.threshold = threshold

    def _random_crop(self, img):
        c, h, w = img.shape
        h = h - self.crop_size
        ymin = np.random.choice(np.arange(h))
        w = w - self.crop_size
        xmin = np.random.choice(np.arange(w))
        img = img[:, ymin:ymin + self.crop_size, xmin:xmin + self.crop_size]
        return img, xmin, ymin

    @staticmethod
    def overlap(img_rect, object_rect):
        """calculates overlap of object bounding box with image bounding box,
        after image has been cropped

        Parameters
        ----------
        img_rect : Rectangle
            image bounding box, represented as a Rectangle (named tuple defined in this module)
        object_rect : Rectangle
            object bounding box, represented as a Rectangle (named tuple defined in this module)

        Returns
        -------
        normal_overlap : float
            between 0 and 1, area of overlap divided by area of object rectangle, i.e. bounding box
        """
        dx = min(img_rect.xmax, object_rect.xmax) - max(img_rect.xmin, object_rect.xmin)
        dy = min(img_rect.ymax, object_rect.ymax) - max(img_rect.ymin, object_rect.ymin)
        if (dx >= 0) and (dy >= 0):
            overlap_area = dx * dy
            object_area = (object_rect.xmax - object_rect.xmin) * (object_rect.ymax - object_rect.ymin)
            normal_overlap = overlap_area / object_area
            return normal_overlap
        else:
            return 0

    def __call__(self, img, target):
        """converts img to Tensor, performs a random crop, and normalizes using ImageNet mean and standard deviation.
        converts target to one-hot labels.

        Parameters
        ----------
        img : PIL.image
        target : dict
            target annotation, .xml annotation file loaded by Elementree and then converted to Python dict
            by VOCDetection. Will be converted into a Tensor of bounding box co-ordinates and integer label
        img_xmin: int
            of image. Used to determine if an object is still within image after cropping.
        img_ymin : int
            of image. Used to determine if an object is still within image after cropping.
        img_size : int
            of image. Used to determine if an object is still within image after cropping.

        Returns
        -------
        tensor : list
            containing lists of [bounding boxes co-ordinates, class name].
            Gets converted to Tensor of size (batch size, 5)
        """
        img = transforms.ToTensor(img)
        img, img_xmin, img_ymin = self._random_crop(img)
        img_bbox = Rectangle(img_xmin, img_ymin, img_xmin + self.crop_size, img_ymin + self.crop_size)

        target_out = []
        objects = target['annotation']['object']
        if type(objects) == dict:  # if just a single object, not a list of them
            objects = [objects]  # wrap in a list so we can iterate over it
        for obj in objects:  # will be a list
            name = obj['name']
            obj_bbox = obj['bndbox']
            obj_bbox = {k: int(v) for k, v in obj_bbox.items()}  # convert string values to int
            obj_bbox = Rectangle(**obj_bbox)
            if self.threshold:
                # only add if overlap is above threshold
                overlap = self.overlap(img_bbox, obj_bbox)
                if overlap >= self.threshold:
                    label_idx = self.class_to_ind[name]
                    row = list(obj_bbox)
                    row.append(label_idx)  # [xmin, ymin, xmax, ymax, label_ind]
                    target_out.append(row)
                else:
                    continue
            else:
                # add no matter what, there's no threshold
                label_idx = self.class_to_ind[name]
                row = list(obj_bbox)
                row.append(label_idx)  # [xmin, ymin, xmax, ymax, label_ind]
                target_out.append(row)

        return img, target  # [[xmin, ymin, xmax, ymax, label_ind], ... ]
