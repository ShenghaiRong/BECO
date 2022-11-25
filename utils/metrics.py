from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from utils.distributed import get_device


class SimSegMetrics():
    """
    Metrics for Semantic Segmentation Task Validation implemented using torch

    It takes all predictions, UPDATE the result on the run and gives the averaged
    result across the full validation set

    It can also be used in training procedure, one can call reset -> update -> 
    get_results to get current performance on a minibatch

    The Y axis confusion matrix is GT, X axis is prediction

    Args:
        n_classes: The number of classes
        ignore_bkg: whether to exclude background in calculation
    """
    def __init__(self, n_classes: int, ignore_bkg: bool=False) -> None:
        self.n_classes = n_classes
        self.ignore_bkg = ignore_bkg
        self.device = get_device()
        self.confusion_matrix = torch.zeros(
            (self.n_classes, self.n_classes),
            device=self.device, requires_grad=False, dtype=torch.long
        )

    def update(self, labels_true:Tensor, labels_pred:Tensor) -> None:
        """labels_true, labels_pred have the shape of B * H * W"""
        for lt, lp in zip(labels_true, labels_pred):
            self.confusion_matrix.add_(self._fast_hist(lt.flatten(), lp.flatten()))

    def _fast_hist(self, label_true:Tensor, label_pred:Tensor) -> np.ndarray:
        """labels_true, labels_pred are 1-D tensor with the shape of HW"""
        # mask out label = 255 and 1 if ignore background
        lb = 1 if self.ignore_bkg else 0
        mask = torch.logical_and((label_true >= lb), (label_true < self.n_classes))
        hist = torch.bincount(
            self.n_classes * label_true[mask] + label_pred[mask],
            minlength=self.n_classes ** 2
        )
        hist = hist.reshape(self.n_classes, self.n_classes)
        return hist

    def all_reduce(self) -> None:
        """
        Used for parallel validation, do all_reduce for confusion_matrix, so all
        threads gets the final confusion_matrix
        """
        dist.all_reduce(self.confusion_matrix, op=dist.ReduceOp.SUM)

    def get_hist(self) -> Tensor:
        """
        return confusion matrix
        """
        return self.confusion_matrix.clone()

    def reset(self) -> None:
        self.confusion_matrix = torch.zeros(
            (self.n_classes, self.n_classes),
            device=self.device, requires_grad=False, dtype=torch.long
        )

    def get_results(self, keys: List[str]) -> Dict[str, float]:
        """
        Returns the metrics based on given keys

        Valid keys are:
            Overall_Acc, mClass_Acc, mPred_Acc, mIoU, Class_Acc,
            Pred_Acc, IoU, GT_Freq, FW_mIoU
        """
        EPS = torch.tensor(1e-5, device=self.device)
        hist = self.confusion_matrix

        gt_sum = hist.sum(dim=1)
        pred_sum = hist.sum(dim=0)
        # exclude classes does not exist in label
        gt_mask = (gt_sum >= 1)
        # exclude classes does not have output
        pred_mask = (pred_sum >= 1)

        diag = torch.diag(hist)

        acc = diag.sum() / hist.sum()
        # `acc_cls_c` is the percentange of class c been correctly predicted
        acc_cls_c = diag / (gt_sum + EPS)
        acc_cls_c[~gt_mask] = 0.
        acc_cls = torch.mean(acc_cls_c[gt_mask])
        # `acc_pred_c` is the percentage of correct predicted class c
        acc_pred_c = diag / (pred_sum + EPS)
        acc_pred_c[~pred_mask] = 0.
        acc_pred = torch.mean(acc_pred_c[pred_mask])
        iou = diag / (gt_sum + hist.sum(dim=0) - diag + EPS)
        mean_iou = torch.mean(iou[gt_mask])
        
        #gt_freq = hist.sum(dim=1) / hist.sum()
        #fwmiou = (freq[freq > 0] * iu[freq > 0]).sum()

        Result_dict = {
                "Overall_Acc": acc.item(),
                "mClass_Acc": acc_cls.item(),
                "mPred_Acc": acc_pred.item(),
                "mIoU": mean_iou.item(),
                "Class_Acc": acc_cls_c.tolist(),
                "Pred_Acc": acc_pred_c.tolist(),
                "IoU": iou.tolist(),
            }

        rt_dict = dict()
        for k in keys:
            rt_dict[k] = Result_dict[k]

        return rt_dict


    