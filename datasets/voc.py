import os
from typing import Callable, Tuple
import numpy as np

import torch
import torch.utils.data as data
import PIL.Image as Image

from .base import BaseDataset
from utils.registers import DATASETS_REG


DATASETS_REG.register_module()
class VOC(BaseDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    num_classes = 21
    classes_map = [
            'background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(
        self,
        is_aug: bool=True,
        mask_dir=None,
        **kwargs
    ):
        super(VOC, self).__init__(**kwargs)

        splits_dir = os.path.join(self.root, 'ImageSets/Segmentation')

        assert os.path.isdir(self.root), f'Dataset not found at location {self.root}'

        self.img_dir = os.path.join(self.root, 'JPEGImages')

        if mask_dir is None:
            mask_dir = os.path.join(self.root, 'SegmentationClassAug')
        assert os.path.exists(mask_dir), "Mask dir not found"

        self.mask_dir = mask_dir

        if is_aug and self.mode == 'train':
            split_f = os.path.join(splits_dir, 'train_aug.txt')
        else:
            split_f = os.path.join(splits_dir, self.mode.rstrip('\n') + '.txt')

        assert os.path.exists(split_f), f"split file not found"
       
        self.file_names = np.loadtxt(split_f, dtype=np.str_)
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img_name = self.file_names[index]
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        mask_path = os.path.join(self.mask_dir, img_name + '.png')
        img = Image.open(img_path).convert('RGB')
        target = Image.open(mask_path)

        if self.transform is not None:
            data = {'img': img, 'label': target}
            data = self.transform(data)
            img, target = data['img'], data['label']

        return {'img': img, 'target': target, 'name': img_name}

    def __len__(self):
        return len(self.file_names)

    def get_info(self):
        msg = f"The number of all {self.mode} images: {self.__len__()}"
        return msg


@DATASETS_REG.register_module("voc_pseu_mask")
class VOCSegmentationPseuMask(VOC):
    def __init__(self, is_aug: bool = True, mask_dir=None, pseumask_dir=None, 
                 **kwargs):
        super().__init__(is_aug, mask_dir, **kwargs)
        self.pseumask_dir = pseumask_dir

    def __getitem__(self, index):
        img_name = self.file_names[index]
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        mask_path = os.path.join(self.mask_dir, img_name + '.png')
        img = Image.open(img_path).convert('RGB')
        target = Image.open(mask_path)
        if self.mode == 'train':
            pseumask_path = os.path.join(self.pseumask_dir, img_name + '.png')
            mask_path = os.path.join(self.mask_dir, img_name + '.png')
            pseumask = Image.open(pseumask_path).convert('1')
            if self.transform is not None:
                data = {'img': img, 'label': target, 'mask': pseumask}
                data = self.transform(data)
                img, target, mask = data['img'], data['label'], data['mask']
            target = target.to(torch.long)
            mask = mask.to(torch.long)
            return {'img': img, 'target': target, 
                    'name': img_name, 'mask': mask}
        else:
            if self.transform is not None:
                data = {'img': img, 'label': target}
                data = self.transform(data)
                img, target = data['img'], data['label']
            target = target.to(torch.long)
            return {'img': img, 'target': target, 'name': img_name}


@DATASETS_REG.register_module('voc_test')
class VOCTest(torch.utils.data.Dataset):
    num_classes = 21
    def __init__(
        self,
        root: str,
        mode: str='train',
        transform: Callable=None,
        **kwargs
    ):
        self.root = root
        self.mode = mode
        self.transform = transform

        if mode == 'train':
            self.transform = None

        if mode == 'test':
            self.root = root + '_test'

        splits_dir = os.path.join(self.root, 'ImageSets/Segmentation')
        assert os.path.isdir(self.root), \
            f'Dataset not found at location {self.root}'
        self.img_dir = os.path.join(self.root, 'JPEGImages')
        split_f = os.path.join(splits_dir, self.mode.rstrip('\n') + '.txt')
        assert os.path.exists(split_f), f"split file not found"
        self.file_names = np.loadtxt(split_f, dtype=np.str_)       

    def __getitem__(self, index):
        img_name = self.file_names[index]
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        img = Image.open(img_path).convert('RGB')
        label = Image.new(mode="RGB", size=(512, 512))

        if self.transform is not None:
            data = {'img': img, 'label': label}
            data = self.transform(data)
            img, label = data['img'], data['label']

        return {'img': img, 'target': label, 'name': img_name}

    def __len__(self):
        return len(self.file_names)

    def get_info(self):
        msg = f"The number of all {self.mode} images: {self.__len__()}"
        return msg


classes_map = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}