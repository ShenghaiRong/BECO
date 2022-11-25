from asyncio import QueueEmpty
from matplotlib.pyplot import cla

from typing import Dict, Tuple, List, Optional
import warnings
from collections.abc import Sequence
import numpy as np
import math
import numbers
import torch
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
import torch.nn as nn
from PIL import ImageFile
import random
import cv2
import albumentations as al

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ToTensor:
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    Note that labels will NOT be normalized to [0, 1].
    """
    def __init__(self, do_label=True) -> None:
        super().__init__()
        self.do_label = do_label

    def __call__(self, data: Dict) -> Dict:
        img, label = data['img'], data['label']
        if self.do_label and label is not None:
            return {'img': F.to_tensor(img), 
                    'label': torch.from_numpy(np.array(label, dtype=np.uint8))}
        elif label is not None:
            return {'img': F.to_tensor(img), 'label': label}
        else:
            return {'img': F.to_tensor(img)}

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize:
    """
    Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, data: Dict) -> Dict:
        img, label = data['img'], data['label']
        if label is not None:
            return {'img': F.normalize(img, self.mean, self.std, self.inplace), 
                    'label': label}
        else:
            return {'img': F.normalize(img, self.mean, self.std, self.inplace)}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize(torch.nn.Module):
    """
    Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).
            In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.

        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
    """
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, data: Dict) -> Dict:
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        img, label = data['img'], data['label']
        
        if label is not None:
            return {
                'img': F.resize(img, self.size, self.interpolation, 
                                self.max_size, self.antialias),
                'label': F.resize(label, self.size, InterpolationMode.NEAREST, 
                                  self.max_size, self.antialias)
            }
        else:
            return {'img': F.resize(img, self.size, self.interpolation, 
                                    self.max_size, self.antialias)}

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + \
            '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)




class ToTensorMask:
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    Note that labels will NOT be normalized to [0, 1].
    """

    def __call__(self, data: Dict) -> Dict:
        img, label, mask = data['img'], data['label'], data['mask']
        if label is not None:
            if mask is not None:
                img = F.to_tensor(img) 
                label = torch.from_numpy(np.array(label, dtype=np.uint8)) 
                mask = F.to_tensor(mask)
                return {'img': img, 'label': label, 'mask': mask}
            else:
                img = F.to_tensor(img) 
                label = torch.from_numpy(np.array(label, dtype=np.uint8)) 
                return {'img': img, 'label': label}
        else:
            return {'img': F.to_tensor(img)}

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToTensorMultiMask:
    def __init__(self, pass_operate=False):
        self.pass_opt = pass_operate

    def __call__(self, data: Dict) -> Dict:
        if self.pass_opt:
            return data
        return {key: F.to_tensor(value) 
                if F.get_image_num_channels(value) == 3
                else torch.from_numpy(np.array(value, dtype=np.uint8))
                for key, value in data.items()}


class ResizeMultiMask(Resize):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, 
                 max_size=None, antialias=None):
        super().__init__(size, interpolation, max_size, antialias)

    def forward(self, data: Dict) -> Dict:
        return {key: F.resize(value, self.size, self.interpolation, 
                                self.max_size, self.antialias)
                if F.get_image_num_channels(value) == 3
                else F.resize(value, self.size, InterpolationMode.NEAREST, 
                                  self.max_size, self.antialias)
                for key, value in data.items()}
        

class NormalizeMask:
    """
    Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, data: Dict) -> Dict:
        img, label, mask = data['img'], data['label'], data['mask']
        if label is not None:
            if mask is not None:                
                img = F.normalize(img, self.mean, self.std, self.inplace)
                return {'img': img, 'label': label, 'mask': mask}
            else:
                img = F.normalize(img, self.mean, self.std, self.inplace) 
                return {'img': img, 'label': label}

        else:
            img = F.normalize(img, self.mean, self.std, self.inplace) 
            return {'img': img}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)






class NormalizeMultiMask(NormalizeMask):
    def __call__(self, data: Dict) -> Dict:
        img = data['img']
        img = F.normalize(img, self.mean, self.std, self.inplace) 
        data['img'] = img
        return data


class RandomResizedCropMask(torch.nn.Module):
    """Crop a random portion of image and resize it to a given size.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.

    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
            img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, data: Dict) -> Dict:
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        img, label, mask = data['img'], data['label'], data['mask']
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if label is not None:
            if mask is not None:
                img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
                label = F.resized_crop(label, i, j, h, w, self.size, InterpolationMode.NEAREST)
                mask = F.resized_crop(mask, i, j, h, w, self.size, InterpolationMode.NEAREST)
                return {'img': img, 'label': label, 'mask': mask}
            else:
                img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
                label = F.resized_crop(label, i, j, h, w, self.size, InterpolationMode.NEAREST)
                return {'img': img, 'label': label}
        else:
            img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
            return {'img': img}

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomResizedCropMultiMask(RandomResizedCropMask):
    def forward(self, data: Dict) -> Dict:
        img = data['img']
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        for key, value in data.items():
            if F.get_image_num_channels(value) == 3:
                new_value = F.resized_crop(value, i, j, h, w, self.size, 
                                           self.interpolation)
            else:
                new_value = F.resized_crop(value, i, j, h, w, self.size, 
                                           InterpolationMode.NEAREST)
            data[key] = new_value
        return data
                
        
    

class RandomHorizontalFlipMask:
    """Horizontally flip the given image randomly with a given probability."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, data: Dict) -> Dict:
        img, label, mask = data['img'], data['label'], data['mask']
        if torch.rand(1) < self.p:
            if label is not None:
                if mask is not None:
                    return {'img': F.hflip(img), 'label': F.hflip(label), 'mask': F.hflip(mask)}
                else:
                    return {'img': F.hflip(img), 'label': F.hflip(label)}
            else:
                return F.hflip(img)
        if label is not None:
            if mask is not None:
                return data
            else:
                return {'img': img, 'label': label}
        else:
            return {'img': img}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomHorizontalFlipMultiMask:
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, data: Dict) -> Dict:
        if torch.rand(1) < self.p:
            return {key: F.hflip(value) for key, value in data.items()}
        else:
            return data

        
def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class DeNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor



def get_al_augmentation():
    return al.Compose([
        # al.RandomResizedCrop(512, 512, scale=(0.2, 1.)), # already resize and random crop in dataset.py
        al.Compose([
            # NOTE: RandomBrightnessContrast replaces ColorJitter
            al.RandomBrightnessContrast(p=1),
            al.HueSaturationValue(p=1),
        ], p=0.8),
        al.ToGray(p=0.2),
        al.GaussianBlur(5, p=0.5),
    ])


def augment_withmask(images, labels, masks):
    """Augments both image and label. Assumes input is a PyTorch tensor with 
       a batch dimension and values normalized to N(0,1)."""

    # Transform label shape: B, C, W, H ==> B, W, H, C
    images = images.clone()
    labels = labels.clone()
    masks = masks.clone()
    labels_are_3d = (len(labels.shape) == 4)
    if labels_are_3d:
        labels = labels.permute(0, 2, 3, 1)

    # Transform each image independently. This is slow, but whatever.
    aug_images, aug_labels = [], []
    aug_masks = []
    for image, label, mask in zip(images, labels, masks):

        # Step 1: Undo normalization transformation, convert to numpy
        denormalize = DeNormalize(MEAN, STD)
        image = denormalize(image) * 255
        image = cv2.cvtColor(image.numpy().transpose(
            1, 2, 0), cv2.COLOR_BGR2RGB).astype(np.uint8)
        #image = cv2.cvtColor(image.numpy().transpose(
        #    1, 2, 0) + IMG_MEAN, cv2.COLOR_BGR2RGB).astype(np.uint8)
        label = label.numpy()  # convert to np
        mask = mask.numpy()  # convert to np

        # Step 2: Perform transformations on numpy images
        #data = aug(image=image, label=label, mask=mask)
        #image, label, mask = data['image'], data['label'], data['mask']
        data = get_al_augmentation()(image=image)
        image = data['image']

        # Step 3: Convert back to PyTorch tensors
        image = torch.from_numpy((cv2.cvtColor(image.astype(
            np.float32), cv2.COLOR_RGB2BGR)).transpose(2, 0, 1)).div(255)
        image = F.normalize(image, MEAN, STD)
        #image = torch.from_numpy((cv2.cvtColor(image.astype(
        #    np.float32), cv2.COLOR_RGB2BGR) - IMG_MEAN).transpose(2, 0, 1))

        label = torch.from_numpy(label)
        mask = torch.from_numpy(mask)
        if not labels_are_3d:
            label = label.long()
            mask = mask.long()

        # Add to list
        aug_images.append(image)
        aug_labels.append(label)
        aug_masks.append(mask)

    # Stack
    images = torch.stack(aug_images, dim=0)
    labels = torch.stack(aug_labels, dim=0)
    masks = torch.stack(aug_masks, dim=0)

    # Transform label shape back: B, C, W, H ==> B, W, H, C
    if labels_are_3d:
        labels = labels.permute(0, 3, 1, 2)
    return images, labels, masks
    










