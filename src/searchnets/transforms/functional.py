"""transforms used with Torch Datasets"""
import collections
import xml.etree.ElementTree as ET
import random

import numpy as np
import torch

__all__ = [
    'random_pad',
    'class_ints_from_xml',
    'largest_class_int_from_xml',
    'onehot_from_class_ints',
    'parse_voc_xml',
    'random_class_int',
    'tensor_from_numpy_scalar',
    'tile',
]

# declare as a constant because also referenced by munge.VSD_results_df
VSD_PAD_SIZE = 500


def random_pad(img, pad_size=VSD_PAD_SIZE):
    """pads VOC image randomly, so that size is constant but location of image is random.
    Makes a tensor filled with zeros of size pad_size and places the image randomly.
    The function ensures the entire image stays within the padding.

    Parameters
    ----------
    img : torch.Tensor
        image converted to torch.Tensor
    pad_size : int
        size of image after padding

    Returns
    -------
    padded : torch.Tensor
        padded to size (pad_size x pad_size)
    """
    c, h, w = img.shape
    if h > pad_size:
        raise ValueError(
            f'height of image {h} is greater than pad size {pad_size}'
        )
    if w > pad_size:
        raise ValueError(
            f'width of image {w} is greater than pad size {pad_size}'
        )
    padded = torch.FloatTensor(c, pad_size, pad_size)
    padded.zero_()
    h_range = pad_size - h
    if h_range > 0:
        y_topleft = torch.randint(h_range, (1,))
    else:
        y_topleft = 0
    w_range = pad_size - w
    if w_range > 0:
        x_topleft = torch.randint(w_range, (1,))
    else:
        x_topleft = 0

    padded[:, y_topleft:y_topleft + h, x_topleft:x_topleft + w] = img

    return padded


def _recurse_nodes(node):
    """helper function that recursively turns nodes from xml tree into Python dicts"""
    node_dict = {}
    children = list(node)
    if children:
        def_dic = collections.defaultdict(list)
        for dc in map(_recurse_nodes, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        node_dict = {
            node.tag:
                {ind: v[0] if len(v) == 1 else v
                 for ind, v in def_dic.items()}
        }
    if node.text:
        text = node.text.strip()
        if not children:
            node_dict[node.tag] = text
    return node_dict


def parse_voc_xml(xml_file):
    """parse .xml annotation of an image from a Pascal VOC dataset.

    Parameters
    ----------
    xml_file : str, Path
        to .xml file that annotations an image from Pascal VOC

    Returns
    -------
    voc_dict : dict
        annotations represented as a Python dictionary
    """
    node = ET.parse(xml_file).getroot()
    voc_dict = _recurse_nodes(node)
    return voc_dict


VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

VOC_CLASS_INT_MAP = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))


def class_ints_from_xml(xml_dict, class_int_map=None):
    """get list of classes in an image from the PascalVOC dataset as a list of ints,
    given the .xml annotation file in a Python dictionary

    Parameters
    ----------
    xml_dict : dict
        that contains .xml annotation.
    class_int_map : dict
        that maps class names from the PascalVOC dataset to integer values.
        Keys are classes in PascalVoc, e.g., 'aeroplane', and values are ints.
        Default is None, in which case searchnets.transforms.functional.VOC_CLASS_INT_MAP is used.

    Returns
    -------
    class_ints : list
        of ints, the classes present in the image.
    """
    class_int_map = class_int_map or VOC_CLASS_INT_MAP

    objects = xml_dict['annotation']['object']
    if type(objects) == dict:  # if just a single object, not a list of them
        objects = [objects]  # wrap in a list so we can iterate over it

    class_ints = []
    for obj in objects:  # will be a list
        name = obj['name']
        class_ints.append(class_int_map[name])
    return class_ints


def random_class_int(class_ints):
    """transform that takes list of classes present in a PascalVOC dataset image,
    and returns one chosen randomly.

    Parameters
    ----------
    class_ints : list
        of ints, corresponding to the classes present in the image.
        As returned by searchnets.transforms.functional.class_ints_from_xml

    Returns
    -------
    random_class_int
    """
    return random.choice(class_ints)


def _size_from_bndbox(bndbox):
    """helper function that determines size of bounding box from Pascal VOC .xml annotation file"""
    bndbox = {k: int(v) for k, v in bndbox.items()}  # cast string to int
    height = bndbox['ymax'] - bndbox['ymin']
    width = bndbox['xmax'] - bndbox['xmin']
    return height * width


def largest_class_int_from_xml(xml_dict, class_int_map=None):
    """get class of largest object in an image from the PascalVOC dataset,
    as determined by the size of its bounding box in the .xml annotation file

    Parameters
    ----------
    xml_dict : dict
        that contains .xml annotation.
    class_int_map : dict
        that maps class names from the PascalVOC dataset to integer values.
        Keys are classes in PascalVoc, e.g., 'aeroplane', and values are ints.
        Default is None, in which case searchnets.transforms.functional.VOC_CLASS_INT_MAP is used.

    Returns
    -------
    class_int : int
        of largest object, as determined by the size of its bounding box
    """
    class_int_map = class_int_map or VOC_CLASS_INT_MAP

    objects = xml_dict['annotation']['object']
    if type(objects) == dict:  # only a single object, just return it
        name = objects['name']
        return class_int_map[name]
    elif type(objects) == list:
        sizes = [_size_from_bndbox(obj['bndbox']) for obj in objects]
        largest_obj_ind = np.argmax(sizes)
        name = objects[largest_obj_ind]['name']
        return class_int_map[name]
    else:
        raise ValueError(
            f'unexpected type in xml_dict: {type(objects)}'
        )


def onehot_from_class_ints(class_ints, n_classes=None):
    """convert list of integers representing classes present in an image from PascalVOC
    to a one-hot vector encoding of the classes present.

    Parameters
    ----------
    class_ints : list
        of ints, classes present in an image
    n_classes : int
        total number of classes present in dataset.
        Default is None, in which case len(searchnets.transforms.functional.VOC_CLASS_INT_MAP) is used.

    Returns
    -------
    onehot : torch.Tensor
    """
    # we don't use functional.one_hot because we want to return an array of all zeros if none of the objects
    # are present; can't pass an array of "nothing" to functional.one_hot to get that output
    n_classes = n_classes or len(VOC_CLASS_INT_MAP)
    onehot = torch.FloatTensor(n_classes)
    onehot.zero_()
    return onehot.scatter_(0, torch.LongTensor(class_ints), 1)


def tensor_from_numpy_scalar(scalar):
    return torch.from_numpy(np.asarray(scalar))


def tile(a, dim, n_tile):
    """tile elements of tensor ``a`` along dimension ``dim`` a specified number of times.
    E.g., to repeat samples consecutively along batch dimension.

    Parameters
    ----------
    a : torch.Tensor
        tensor which should have some dimension tiled
    dim : int
        dimension to tile
    n_tile : int
        number of times each element in that dimension should be tiled

    Returns
    -------
    a_tiled : torch.Tensor

    Examples
    --------
    >>> t = torch.tensor([[1, 2, 3], [4, 4, 4]])
    >>> tile(t, 0, 3)
    tensor([[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [4, 4, 4],
            [4, 4, 4],
            [4, 4, 4]])

    Notes
    -----
    adapted from https://discuss.pytorch.org/t/repeat-examples-along-batch-dimension/36217/4
    """
    dim_init_size = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([dim_init_size * np.arange(n_tile) + i for i in range(dim_init_size)])
    )
    return torch.index_select(a, dim, order_index)
