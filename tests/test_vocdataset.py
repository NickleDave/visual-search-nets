from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from searchnets.datasets import VOCDetection, DATASET_YEAR_DIC
from searchnets.utils.transforms import VOCTransform, Rectangle

YEAR = '2012'
VOC_DATASET_ROOT = Path('~/Documents/data/voc')
VOC_DATASET_ROOT = VOC_DATASET_ROOT.expanduser()
VOC_DATASET_ROOT = VOC_DATASET_ROOT.joinpath(DATASET_YEAR_DICT[YEAR]['base_dir'])
IMAGE_SET = 'trainval'


class TestVOCDataset(unittest.TestCase):
    def setUp(self):
        splits_dir = VOC_DATASET_ROOT.joinpath('ImageSets/Main')
        split_f = splits_dir.joinpath(IMAGE_SET.rstrip('\n') + '.txt')
        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        annotation_dir = VOC_DATASET_ROOT.joinpath('Annotations')
        self.annotations = [annotation_dir.joinpath(x + ".xml") for x in file_names]

    def test_overlap(self):
        img_bbox = Rectangle(0, 0, 224, 224)
        obj_bbox = Rectangle(xmin=34, ymin=11, xmax=448, ymax=293)
        normal_overlap = VOCAnnotationTransform.overlap(img_bbox, obj_bbox)
        self.assertTrue(
            type(normal_overlap) == float
        )

    def test_VOCAnnotationTransform_no_threshold(self):
        voc_annot_transform = VOCTransform()
        self.assertTrue(
            voc_annot_transform.threshold is None
        )
        index = 0
        root = ET.parse(self.annotations[index]).getroot()
        target = VOCDetection.parse_voc_xml(root)
        self.assertTrue(
            type(target) == dict
        )
        target = voc_annot_transform(target)
        self.assertTrue(
            type(target) == list
        )
        self.assertTrue(
            all([type(el) == int for el in target[0]])
        )

    def test_VOCAnnotationTransform_with_overlap_above_threshold(self):
        voc_annot_transform = VOCTransform(threshold=0.5)
        self.assertTrue(
            voc_annot_transform.threshold == 0.5
        )
        index = 1
        root = ET.parse(self.annotations[index]).getroot()
        target_before = VOCDetection.parse_voc_xml(root)
        self.assertTrue(
            type(target_before) == dict
        )
        target_after = voc_annot_transform(target_before, img_xmin=34, img_ymin=11, img_size=224)
        self.assertTrue(
            type(target_after) == list
        )
        self.assertTrue(
            all([type(el) == int for el in target_after[0]])
        )

    def test_VOCAnnotationTransform_with_overlap_below_threshold(self):
        voc_annot_transform = VOCTransform(threshold=0.5)
        self.assertTrue(
            voc_annot_transform.threshold == 0.5
        )
        index = 0
        root = ET.parse(self.annotations[index]).getroot()
        target = VOCDetection.parse_voc_xml(root)
        self.assertTrue(
            type(target) == dict
        )
        target = voc_annot_transform(target, img_xmin=0, img_ymin=0, img_size=224)
        self.assertTrue(
            type(target) == list
        )
        self.assertTrue(
            not target  # empty, because overlap of image bounding box and object bounding box is below threshold
        )


if __name__ == '__main__':
    unittest.main()
