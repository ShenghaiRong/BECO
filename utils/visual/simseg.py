from typing import Union, List
import math

import numpy as np
import torch
import torchvision.transforms.functional as trans_F
import PIL.Image as Image
import matplotlib.pyplot as plt
import seaborn as sns
from datasets.transforms.transform import Resize, DeNormalize



"""
SimSeg visualization help functions
"""

def vis_data(
    image: torch.Tensor, 
    label: torch.Tensor,
    img_name: str=None,
    img_denormalize=True,
    cmap='VOC'):
    vis_images = image.clone().cpu()
    vis_labels = label.clone().cpu()
    if len(image.size()) == 3:
        vis_images = vis_images.unsqueeze(0)
        vis_labels = vis_labels.unsqueeze(0)
    for i in range(vis_images.size(0)):
        vis_image = vis_images[i]
        vis_label = vis_labels[i]
        if img_denormalize:
            denormalize = DeNormalize()
            vis_image = denormalize(vis_image) * 255
        vis_image = vis_image.numpy().transpose(1, 2, 0).astype(np.uint8)
        vis_image = Image.fromarray(vis_image)
        vis_label = vis_label.numpy()
        w, h = vis_label.shape
        color_map = get_cmap(cmap)
        # change 255 to the last color of color map
        vis_label[vis_label == 255] = len(color_map.cmap) - 1
        # select color based on label
        vis_label = np.take(color_map.cmap, vis_label, axis=0)
        # concatenate, and reshape it to an image
        vis_label = np.concatenate(vis_label, axis=0)
        vis_label = vis_label.reshape((w, h, 3))
        vis_label = Image.fromarray(vis_label)
        save_dir = f'./data/vis_image/{img_name}{i}'
        vis_image.save(f'{save_dir}.jpg')
        vis_label.save(f'{save_dir}.png')
    return True

def vis_mask(masks: torch.Tensor, img_name: str):
    vis_masks = masks.clone().cpu().squeeze(1)
    for i in range(vis_masks.size(0)):
        mask = vis_masks[i].numpy().astype(np.uint8) * 255
        mask = Image.fromarray(mask)
        save_dir = f'./data/vis_image/{img_name}_mask{i}.png'
        mask.save(save_dir)


def vis_label(
    label,
    img_name: str=None,
    output_dir: str=None,
    cmap='VOC'):
    vis_label = label
    for i in range(1):
        w, h = vis_label.shape
        color_map = get_cmap(cmap)
        # change 255 to the last color of color map
        vis_label[vis_label == 255] = len(color_map.cmap) - 1
        # select color based on label
        vis_label = np.take(color_map.cmap, vis_label, axis=0)
        # concatenate, and reshape it to an image
        vis_label = np.concatenate(vis_label, axis=0)
        vis_label = vis_label.reshape((w, h, 3))
        vis_label = Image.fromarray(vis_label)
        save_dir = f'{output_dir}/{img_name}'
        vis_label.save(f'{save_dir}.png')
    return True



    
    


def colorize_label(
    image: Union[torch.Tensor, Image.Image, np.ndarray], color_map = None
) -> torch.Tensor:
    """
    This function colorize labels or predictions based on given color_map
    """
    # change label to 1-D np array
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.numpy()
    w, h = image.shape
    # change 255 to the last color of color map
    image[image == 255] = len(color_map.cmap) - 1
    # select color based on label
    vis_image = np.take(color_map.cmap, image, axis=0)
    # concatenate, and reshape it to an image
    vis_image = np.concatenate(vis_image, axis=0)
    vis_image = vis_image.reshape((w, h, 3))
    vis_image = trans_F.to_tensor(vis_image)

    return vis_image


def blend_image_label(
    image: Union[torch.Tensor, Image.Image, np.ndarray], 
    label: Union[torch.Tensor, Image.Image, np.ndarray],
    color_map = None,
    alpha: float=0.8
) -> torch.Tensor:
    """
    This function takes an image and an label, first colorize label based on 
        color_map and put them together for visualization

    Args:
        image: shape of H * W
        label: same size as image
        color_map: an ColorMap object
    """
    # change image to PIL Imgage
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        image = trans_F.to_pil_image(image).convert('RGB')
    # change label to 1-D np array
    if isinstance(label, Image.Image):
        label = np.array(label)
    elif isinstance(label, torch.Tensor):
        label = label.numpy()

    w, h = label.shape
    # change 255 to the last color of color map
    label[label == 255] = len(color_map.cmap) - 1
    # select color based on label
    vis_label = np.take(color_map.cmap, label, axis=0)
    # concatenate, and reshape it to an image
    vis_label = np.concatenate(vis_label, axis=0)
    vis_label = vis_label.reshape((w, h, 3))
    vis_label= Image.fromarray(vis_label)
    vis_result = Image.blend(image, vis_label, alpha = alpha)
    # tensorboard does not support PIL Image
    vis_result = trans_F.to_tensor(vis_result)

    return vis_result


def get_cmap(cmap: str):
    if cmap == 'VOC':
        return VOCCmap()
    elif cmap == "ADE":
        return ADECmap()
        

class ColorMap():
    """
    This class generate color map for visualization
    """
    def __init__(self, num_colors):
        self.cmap = self.init_cmap(N = num_colors)

    def init_cmap(self, N=256, normalized=False) -> List[np.ndarray]:
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap/255 if normalized else cmap
        return cmap


class VOCCmap(ColorMap):
    def __init__(self):
        super().__init__(num_colors = 22)
        self.cmap[21] = np.array([255, 255, 255])
        self.index2name = (
            'background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        )


class ADECmap(ColorMap):
    def __init__(self):
        super().__init__(num_colors = 152)
        self.cmap[151] = np.array([255, 255, 255])

VOC_index2name = [
            'background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]