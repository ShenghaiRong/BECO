import os
from typing import Callable, List, Tuple

import torch
import torch.utils.data as data
import PIL.Image as Image

from .base_incremental import BaseIncremental
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



@DATASETS_REG.register_module("COCO14")
class COCOSegmentation14(data.Dataset):
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
    num_classes=81
    def __init__(
        self,
        root: str,
        train: bool=True,
        transform: Callable=None,
        no_label: bool=False,
        rt_index: bool=False
    ) -> None:
        self.num_classes=81
        self.root = root
        self.transform = transform
        self.no_label = no_label
        self.rt_index = rt_index
        self.image_set = "train" if train else "val"
        
        assert os.path.exists(self.root), f'Dataset not found at location {self.root}'

        if self.image_set == 'train':
            self.img_dir = os.path.join(self.root, "train2014")
            self.annot_dir = os.path.join(self.root, "mask/train2014")
            assert os.path.exists(self.img_dir), f"Train images not found{self.img_dir}"
            assert os.path.exists(self.annot_dir), f"Train image annotations not found{self.annot_dir}"
        else:
            self.img_dir = os.path.join(self.root, "val2014")
            self.annot_dir = os.path.join(self.root, "mask/val2014")
            assert os.path.exists(self.img_dir), f"Val images not found{self.img_dir}"
            assert os.path.exists(self.annot_dir), f"Val image annotations not found{self.annot_dir}"

        # sort files to make
        self.img_name = os.listdir(self.img_dir)
        self.img_name.sort()
        # print(self.img_name[0])
        # annotations has the same filename as image
        self.imgs = [(
            os.path.join(self.img_dir, n),
            os.path.join(self.annot_dir, str(int(n[:-4].split("_")[2]))+".png")
            ) for n in self.img_name
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

        # tar = tar.to(torch.long)

        if self.rt_index:
            return img, tar, self.img_name[index][:-4]
        else:
            return img, tar

    def __len__(self):
        return len(self.imgs)


@DATASETS_REG.register_module("COCO14_pseu")
class COCOSegmentation14Pseu(COCOSegmentation14):
    def __init__(
        self,
        root: str,
        train: bool=True,
        transform: Callable=None,
        no_label: bool=False,
        rt_index: bool=False,
        mask_dir: str=None,
        pseumask_dir: str=None,**kwargs
    ) -> None:
        super().__init__(root,train,transform,no_label,rt_index)
        self.mask_dir=mask_dir
        self.pseumask_dir=pseumask_dir
        if self.mask_dir:
            self.imgs = [(
                os.path.join(self.img_dir, n),
                os.path.join(self.mask_dir, n[:-3] + "png"),
                os.path.join(self.pseumask_dir, n[:-3] + "png"),
                ) for n in self.img_name
            ]

    def __getitem__(self, index):
        img = Image.open(self.imgs[index][0]).convert('RGB')
        target = Image.open(self.imgs[index][1])
        img_name=self.img_name[index][:-4]
        if self.image_set == "train":
            pseumask = Image.open(self.imgs[index][2]).convert('1')
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


    








