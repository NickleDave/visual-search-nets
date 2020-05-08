from . import functional as F


__all__ = [
    'ClassIntsFromXml',
    'LargestClassIntFromXml',
    'OneHotFromClassInts',
    'ParseVocXml',
    'RandomClassInt',
    'RandomPad',
    'TensorFromNumpyScalar',
]


class RandomPad:
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
    def __init__(self, pad_size):
        self.pad_size = pad_size

    def __call__(self, img):
        return F.random_pad(img, self.pad_size)


class ParseVocXml:
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
    def __init__(self):
        pass

    def __call__(self, xml_file):
        return F.parse_voc_xml(xml_file)


class ClassIntsFromXml:
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
    def __init__(self, class_int_map=F.VOC_CLASS_INT_MAP):
        self.class_int_map = class_int_map

    def __call__(self, xml_dict):
        return F.class_ints_from_xml(xml_dict, self.class_int_map)


class RandomClassInt:
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
    def __init__(self):
        pass

    def __call__(self, class_ints):
        return F.random_class_int(class_ints)


class LargestClassIntFromXml:
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
    def __init__(self, class_int_map=F.VOC_CLASS_INT_MAP):
        self.class_int_map = class_int_map

    def __call__(self, xml_dict):
        return F.largest_class_int_from_xml(xml_dict, self.class_int_map)


class OneHotFromClassInts:
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
    def __init__(self, n_classes=len(F.VOC_CLASS_INT_MAP)):
        self.n_classes = n_classes

    def __call__(self, class_ints):
        return F.onehot_from_class_ints(class_ints, self.n_classes)


class TensorFromNumpyScalar:
    """convert a scalar value from a numpy array to a torch Tensor.

    Parameters
    ----------
    scalar : numpy scalar value
        e.g., numpy.int64

    Returns
    -------
    tensor : torch.Tensor
        with one element
    """
    def __init__(self):
        pass

    def __call__(self, scalar):
        return F.tensor_from_numpy_scalar(scalar)
