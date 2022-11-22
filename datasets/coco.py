import os
from typing import Callable, List, Tuple

import torch
import torch.utils.data as data
import PIL.Image as Image

from utils.registers import DATASETS_REG


@DATASETS_REG.register_module("COCO")
class COCOSegmentation(data.Dataset):
    """
    COCO Segmentation Datasets

    Args:
        root (string): Root directory of the VOC Dataset.
        train (bool): Select the image_set to use, set True to use training set,
            or False to use val set
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        no_label (bool): if set to True, return all-zero labels
        rt_index (bool): if set to True, additionally return data index
    """

    def __init__(
        self,
        root: str,
        train: bool=True,
        transform: Callable=None,
        no_label: bool=False,
        rt_index: bool=False
    ) -> None:
    
        self.root = root
        self.transform = transform
        self.no_label = no_label
        self.rt_index = rt_index
        self.image_set = "train" if train else "val"
        
        assert os.path.exists(self.root), f'Dataset not found at location {self.root}'

        if self.image_set == 'train':
            img_dir = os.path.join(self.root, "images/train2017")
            annot_dir = os.path.join(self.root, "annotations/stuff_train2017_pixelmaps")
            assert os.path.exists(img_dir), f"Train images not found{img_dir}"
            assert os.path.exists(annot_dir), f"Train image annotations not found{annot_dir}"
        else:
            img_dir = os.path.join(self.root, "images/val2017")
            annot_dir = os.path.join(self.root, "annotations/stuff_val2017_pixelmaps")
            assert os.path.exists(img_dir), f"Val images not found{img_dir}"
            assert os.path.exists(annot_dir), f"Val image annotations not found{annot_dir}"

        # sort files to make
        img_name = os.listdir(img_dir)
        img_name.sort()
        # annotations has the same filename as image
        self.imgs = [(
            os.path.join(img_dir, n),
            os.path.join(annot_dir, n[:-3] + "png")
            ) for n in img_name
        ]
            
    def __getitem__(self, index):
        """
        NOTE: In this function, index is additionally returned

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.imgs[index][0]).convert('RGB')
        if self.no_label:
            img = self.transform(img)
            tar = torch.zeros_like(img, dtype=torch.long)
        else:
            tar = Image.open(self.imgs[index][1])
            img, tar = self.transform(img, tar)

        tar = tar.to(torch.long)

        if self.rt_index:
            return img, tar, index
        else:
            return img, tar

    def __len__(self):
        return len(self.imgs)


@DATASETS_REG.register_module("COCO_Inc")
class COCOSegmentationSubset(data.Dataset):
    """
    Create a subset of COCO dataset based on given indexes

    Args:
        - index: (List[int]) The subset data index
    """

    def __init__(
        self,
        root: str,
        index: List[int],
        train: bool=True,
        transform: Callable=None,
        no_label: bool=False,
        rt_index: bool=False
    ) -> None:
        
        self.root = root
        self.transform = transform
        self.no_label = no_label
        self.rt_index = rt_index
        self.image_set = "train" if train else "val"
        self.imgs = []
        
        assert os.path.exists(self.root), f'Dataset not found at location {self.root}'

        if self.image_set == 'train':
            img_dir = os.path.join(self.root, "images/train2017")
            annot_dir = os.path.join(self.root, "annotations/stuff_train2017_pixelmaps")
            assert os.path.exists(img_dir), f"Train images not found{img_dir}"
            assert os.path.exists(annot_dir), f"Train image annotations not found{annot_dir}"
        else:
            img_dir = os.path.join(self.root, "images/val2017")
            annot_dir = os.path.join(self.root, "annotations/stuff_val2017_pixelmaps")
            assert os.path.exists(img_dir), f"Val images not found{img_dir}"
            assert os.path.exists(annot_dir), f"Val image annotations not found{annot_dir}"

        # sort files to make
        img_name = os.listdir(img_dir)
        img_name.sort()
        # annotations has the same filename as image
        for i in index:
            self.imgs.append((
                os.path.join(img_dir, img_name[i]),
                os.path.join(annot_dir, img_name[i])
            ))

    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        img = Image.open(self.imgs[index][0]).convert('RGB')
        if self.no_label:
            img = self.transform(img)
            tar = torch.zeros(img.shape[1:], dtype=torch.long)
        else:
            tar = Image.open(self.imgs[index][1])
            img, tar = self.transform(img, tar)

        if self.rt_index:
            return img, tar, index
        else:
            return img, tar

    def __len__(self):
        return len(self.imgs)
