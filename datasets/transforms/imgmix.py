import numpy as np
import torch
import random
from typing import List, Dict
from torch import Tensor
import cv2


def getBoundry(mask: Tensor, size=2, iterations=1) -> Tensor:
    if torch.unique(mask).size(0) == 1:
        bdry = torch.zeros_like(mask).unsqueeze(0) 
        inside = torch.zeros_like(mask).unsqueeze(0)
        return bdry, inside

    mask_np = mask.squeeze(0).numpy().astype(np.uint8)

    pixels = 2 * size + 1    ## Double and plus 1 to have an odd-sized kernel
    kernel = np.ones((pixels, pixels), np.uint8)   ## Pixel of extension I get
    erode_mask = cv2.erode(mask_np*255, kernel, iterations=iterations)  ## How many erosion do you expect
    erode_mask = (erode_mask > 254).astype(np.uint8)
    dilate_mask = cv2.dilate(mask_np*255, kernel, iterations=iterations)
    dilate_mask = (dilate_mask > 254).astype(np.uint8)

    bdry = dilate_mask - erode_mask
    bdry = torch.from_numpy(bdry)
    bdry = bdry.type_as(mask).unsqueeze(0)
    inside = torch.from_numpy(erode_mask)
    inside = inside.type_as(mask).unsqueeze(0)
    return bdry, inside

def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N


def random_cls_mask(label, ignore_index=[255]):
    query_cls = []
    img_class = torch.unique(label)
    for idx in ignore_index:
        img_class = img_class[img_class != idx]
    num_class = len(img_class)
    class_rd = np.random.choice(img_class.numpy(), 
                                int((num_class+num_class%2)/2), replace=False)
    for k in range(len(class_rd)):
        query_cls.append(class_rd[k])

    classes = torch.Tensor(query_cls).long()
    mixmask = generate_class_mask(label.squeeze(0), classes)
    return mixmask



def _mix_one_withmask(data_list1: List, data_list2: List, ignore_bg=False): 
    ignore_index = [255]
    if ignore_bg:
        ignore_index.append(0)
    label2 = data_list2[1]
    alpha = random_cls_mask(label2, ignore_index)
    new_data = (down*(1-alpha) + up*alpha 
                for down, up in zip(data_list1, data_list2))
    return new_data

def imgmix_one_withmask_bdry(data_list1, data_list2, ignore_bg, size, p):
    if random.random() < p:
        img, label, mask = _mix_one_withmask(data_list1, data_list2, ignore_bg)
        bdry = torch.zeros_like(mask)
        inside = torch.zeros_like(mask)
    else:
        img, label, mask = data_list1[0], data_list1[1], data_list1[2]
        bdry = torch.zeros_like(mask)
        inside = torch.zeros_like(mask)
    return img, label, mask, bdry, inside


def imgmix_multi_withmask_bdry(imgs, labels, masks, size=1,
                               ignore_bg=False, p=1) -> Dict:
    # p=0: no mix; p=1: all mix; 0<p<1: random mix with p

    mid_num = imgs.size(0)
    assert mid_num % 2 == 0, 'batch_size must be a multiple of 2'
    mid_num = mid_num // 2
    if len(labels.size()) == 3:
        labels = labels.unsqueeze(1)
    img1 = imgs[:mid_num]
    img2 = imgs[mid_num:]
    label1 = labels[:mid_num]
    label2 = labels[mid_num:]
    mask1 = masks[:mid_num]
    mask2 = masks[mid_num:]
    mix_imgs = []
    mix_labels = []
    mix_masks = []
    mix_bdrys = []
    mix_insides = []
    mix_isbdrys = []
    for i in range(mid_num):
        data_list1 = [img1[i], label1[i], mask1[i]]
        data_list2 = [img2[i], label2[i], mask2[i]]


        img, label, mask, bdry, inside = imgmix_one_withmask_bdry(data_list1, 
                                                                  data_list2, 
                                                          ignore_bg, size, p)
        if torch.sum(bdry) == 0:
            isbdry = torch.zeros_like(bdry)
        else:
            isbdry = torch.ones_like(bdry)
        mix_imgs.append(img.unsqueeze(0))
        mix_labels.append(label)
        mix_masks.append(mask)
        mix_bdrys.append(bdry)
        mix_insides.append(inside)
        mix_isbdrys.append(isbdry)

        img, label, mask, bdry, inside = imgmix_one_withmask_bdry(data_list2, 
                                                                  data_list1, 
                                                          ignore_bg, size, p)
        if torch.sum(bdry) == 0:
            isbdry = torch.zeros_like(bdry)
        else:
            isbdry = torch.ones_like(bdry)
        mix_imgs.append(img.unsqueeze(0))
        mix_labels.append(label)
        mix_masks.append(mask)
        mix_bdrys.append(bdry)
        mix_insides.append(inside)
        mix_isbdrys.append(isbdry)
    mix_imgs = torch.cat(mix_imgs, dim=0)
    mix_labels = torch.cat(mix_labels, dim=0)
    mix_masks = torch.cat(mix_masks, dim=0)
    mix_bdrys = torch.cat(mix_bdrys, dim=0)
    mix_insides = torch.cat(mix_insides, dim=0)
    mix_isbdrys = torch.cat(mix_isbdrys, dim=0)
    if len(mix_labels.size()) == 4:
        mix_labels = mix_labels.squeeze(1)
    return mix_imgs, mix_labels, mix_masks, mix_bdrys, mix_insides, mix_isbdrys


def _mix_one_withmaskol_bdry(data_list1: List, data_list2: List, size=2,
                               ignore_bg=False): 
    ignore_index = [255]
    if ignore_bg:
        ignore_index.append(0)

    mask2 = data_list2[4]
    label_ol2 = data_list2[3]
    label2 = torch.where(mask2.type(torch.bool), label_ol2, 255)
    #label2 = label_ol2 # original classmix
    alpha = random_cls_mask(label2, ignore_index)
    bdry, inside = getBoundry(alpha, size)
    new_data = (down*(1-alpha) + up*alpha 
                for down, up in zip(data_list1, data_list2))
    return *new_data, bdry, inside

def imgmix_one_withmaskol_bdry(data_list1, data_list2, ignore_bg, size, p):
    if random.random() < p:
        img, label_off, mask_off, label_ol, mask_ol, bdry, inside = \
            _mix_one_withmaskol_bdry(data_list1, data_list2, size, ignore_bg)
        label = label_ol
        mask = mask_ol
    else:
        img, label, mask = data_list1[0], data_list1[1], data_list1[2]
        bdry = torch.zeros_like(mask)
        inside = torch.zeros_like(mask)

    return img, label, mask, bdry, inside


def imgmix_multi_withmaskol_bdry(imgs, labels, masks, labels_ol, masks_ol, 
                                 size=2, ignore_bg=False, p=1.0) -> Dict:
    # p=0: no mix; p=1: all mix; 0<p<1: random mix with p

    mid_num = imgs.size(0)
    assert mid_num % 2 == 0, 'batch_size must be a multiple of 2'
    mid_num = mid_num // 2
    if len(labels.size()) == 3:
        labels = labels.unsqueeze(1)
    if len(labels_ol.size()) == 3:
        labels_ol = labels_ol.unsqueeze(1)
    img1 = imgs[:mid_num]
    img2 = imgs[mid_num:]
    label1 = labels[:mid_num]
    label2 = labels[mid_num:]
    mask1 = masks[:mid_num]
    mask2 = masks[mid_num:]
    label_ol1 = labels_ol[:mid_num]
    label_ol2 = labels_ol[mid_num:]
    mask_ol1 = masks_ol[:mid_num]
    mask_ol2 = masks_ol[mid_num:]

    mix_imgs = []
    mix_labels = []
    mix_masks = []
    mix_bdrys = []
    mix_insides = []
    mix_isbdrys = []
    for i in range(mid_num):
        data_list1 = [img1[i], label1[i], mask1[i], label_ol1[i], mask_ol1[i]]
        data_list2 = [img2[i], label2[i], mask2[i], label_ol2[i], mask_ol2[i]]


        img, label, mask, bdry, inside = imgmix_one_withmaskol_bdry(data_list1, 
                                                                  data_list2, 
                                                          ignore_bg, size, p)
        if torch.sum(bdry) == 0:
            isbdry = torch.zeros_like(bdry)
        else:
            isbdry = torch.ones_like(bdry)
        mix_imgs.append(img.unsqueeze(0))
        mix_labels.append(label)
        mix_masks.append(mask)
        mix_bdrys.append(bdry)
        mix_insides.append(inside)
        mix_isbdrys.append(isbdry)

        img, label, mask, bdry, inside = imgmix_one_withmaskol_bdry(data_list2, 
                                                                  data_list1, 
                                                          ignore_bg, size, p)
        if torch.sum(bdry) == 0:
            isbdry = torch.zeros_like(bdry)
        else:
            isbdry = torch.ones_like(bdry)
        mix_imgs.append(img.unsqueeze(0))
        mix_labels.append(label)
        mix_masks.append(mask)
        mix_bdrys.append(bdry)
        mix_insides.append(inside)
        mix_isbdrys.append(isbdry)
    mix_imgs = torch.cat(mix_imgs, dim=0)
    mix_labels = torch.cat(mix_labels, dim=0)
    mix_masks = torch.cat(mix_masks, dim=0)
    mix_bdrys = torch.cat(mix_bdrys, dim=0)
    mix_insides = torch.cat(mix_insides, dim=0)
    mix_isbdrys = torch.cat(mix_isbdrys, dim=0)
    if len(mix_labels.size()) == 4:
        mix_labels = mix_labels.squeeze(1)
    return mix_imgs, mix_labels, mix_masks, mix_bdrys, mix_insides, mix_isbdrys