from abc import abstractmethod
from typing import Dict

from torch import Tensor

from .base import BaseMethod
import utils
from utils.registers import DATASETS_REG
from utils.metrics import SimSegMetrics
from utils.buffer import RuntimeBuffer


class BaseSimSeg(BaseMethod):
    """
    A abstract class for simantic segmentation with basic functions

    NOTE: metric is placed in model class to make updating metric more convenient
        than in a hook.
    """
    def __init__(self, config: Dict, nets_dict: Dict) -> None:
        super().__init__(config)
        self.init_net(nets_dict)
        self.get_class_number()
        # Default SimSeg metric
        self.metric = SimSegMetrics(self.num_classes)
        # To trigger training metric update
        self.metric_update_trigger = False
        self.epoch = 0
        self.loss_buffer = RuntimeBuffer(max_length=10)

    def init_net(self, nets_dict: Dict):
        """register the network to be used in this method"""
        self.nets = utils.distributed.get_ddp_net(nets_dict)

    def get_class_number(self):
        self.num_classes = DATASETS_REG.get(self.config.datasets.type).num_classes

    def update_metric_train(self, logits: Tensor, label: Tensor):
        if self.metric_update_trigger:
            self._update_metric(logits, label)

    def update_metric_val(self, logits: Tensor, label: Tensor):
        self._update_metric(logits, label)

    def _update_metric(self, logits: Tensor, label: Tensor):
        preds = logits.detach().max(dim=1)[1]
        self.metric.update(label, preds)

    def update_loss(self, loss_dict: Dict):
        self.loss_buffer.update('train_loss', loss_dict)

    def get_loss(self):
        return self.loss_buffer.get_average('train_loss')

    def before_run(self):
        pass

    def after_run(self):
        pass

    def before_train_epoch(self):
        pass

    def after_train_epoch(self):
        self.epoch += 1
        pass

    def before_train_iter(self):
        pass

    def after_train_iter(self):
        pass

    def before_val(self):
        pass

    def after_val(self):
        pass

    def get_logging_path(self, logging_path):
        self.logging_path = logging_path
