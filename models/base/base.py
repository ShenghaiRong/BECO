from abc import ABC, abstractmethod
from typing import Dict, Any

import torch

from utils.distributed import get_device


class BaseMethod(ABC):
    # The global config should be passed here
    def __init__(self, config: Dict) -> None:

        self.config = config
        # method specific configs
        self.method_configs = {}
        # contains attributes name of this method, will be saved into ckpt
        self.method_vars = list()
        # logger should be initialized in children classes if needed
        self.logger = None
        self.device = get_device()

    def state_dict(self) -> Dict[str, Any]:
        """
        Return a dict contains method specific variables that should be saved to checkpoint
        """
        state_dict = dict()
        for k in self.method_vars:
            state_dict[k] = getattr(self, k)
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load method specific variables from checkpoint
        """
        for k, v in state_dict.items():
            setattr(self, k, v)

    def scatter(self, data):
        """
        Put training datas to GPU

        Args:
            data: Tensor or Sequence[Tensor]
        """
        if isinstance(data, list) or isinstance(data, tuple):
            return [item.cuda(device=self.device, non_blocking=True)
                    for item in data
            ]
        elif isinstance(data, torch.Tensor):
            return data.cuda(device=self.device, non_blocking=True)
        else:
            raise Exception


    def scatter2cpu(self, data):
        """
        Put training datas to CPU

        Args:
            data: Tensor or Sequence[Tensor]
        """
        if isinstance(data, list) or isinstance(data, tuple):
            return [item.cpu() for item in data]
        elif isinstance(data, torch.Tensor):
            return data.cpu()
        else:
            raise Exception

    def set_up_amp(self, is_amp: bool=False):
        self.is_amp = is_amp

    @abstractmethod
    def init_net(self):
        pass

    @abstractmethod
    def init_loss(self):
        pass

    @abstractmethod
    def train_step(self, data):
        pass

    @abstractmethod
    def val_step(self, data):
        pass

    @abstractmethod
    def forward(self, data):
        pass