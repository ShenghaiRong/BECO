from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List

import torch
from utils.distributed import get_dist_info

import hooks
from utils import distributed
from utils import buffer


class BaseRunner(metaclass=ABCMeta):
    """
    Base Runner class

    Args:
        args: the CLI arguments
        model: The model object
        optimizer: optimizer object
        scheduler: scheduler object
        dataloaders: A dict of dataloaders
        samplers: A dict of samplers for dataloaders
        workflow: workflow control sequence
    """
    def __init__(
        self,
        args,
        model,
        optimizer,
        scheduler,
        dataloaders: Dict,
        samplers: Dict,
        workflow: Dict,
        **kwargs
    ) -> None:
        
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.samplers = samplers

        self.net_dict = distributed.skip_module_for_dict(self.model.nets)
        self.rank = get_dist_info()[0]
        self.epoch = 0
        self.iter = 0
        self.inner_iter = 0
        self._workflow = workflow

        # Buffer to store runtime information. Usually a hook exchange data with
        # other hooks or objects asynchronously using this buffer
        self.buffer = buffer.SimpleBuffer()

        # The attributes name listed in this array will be saved to ckpt. A children
        # class can append more elements to this list
        self.stat_dict_keys = ['epoch', 'iter', 'inner_iter']

        # A logger to output runtime information. In this framework, `TqdmLoggerHook`
        # can be used for better experience. Logger should be initialized in children class
        self.logger = None

        #self.register_hook(config.hooks)

        self._hook_dict = hooks.get_hooks(self.args.logging_path, 
                                          self.args.ckpt,
                                          self.args.amp)

        self._init(**kwargs)

    @abstractmethod
    def _init(self):
        """Further init operations for children classes"""
        pass

    # Main functions
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def val(self):
        pass
    
    @abstractmethod
    def run(self):
        """Entrance to start runner"""
        pass

    @torch.no_grad()
    def test(self):
        """Perform full testing on datasets['test']"""
        self.change_net_val()
        self.is_train = False
        self.logger.write("Start testing...")

        #self.call_hook('before_test')
        self._hook_dict['RestoreCkptHook'].load_ckpt(self)
        if self._hook_dict['TqdmLoggerHook'] is not None:
            self._hook_dict['TqdmLoggerHook'].init_bar_iter_test(self)

        for i, data in enumerate(self.dataloaders['test']):
            #self.call_hook('before_test_iter')
            self.test_iter(data)
            self.inner_iter = i
            #self.call_hook('after_test_iter')
            if self._hook_dict['TqdmLoggerHook'] is not None:
                self._hook_dict['TqdmLoggerHook'].update_bar_iter(self)

        #self.call_hook('after_test')
        self._hook_dict['MetricHook'].get_test_metric(self)
        if self._hook_dict['TqdmLoggerHook'] is not None:
            self._hook_dict['TqdmLoggerHook'].log_test(self)
        if self._hook_dict['TBLoggerHook'] is not None:
            self._hook_dict['TBLoggerHook'].log_test(self)


    
    def test_iter(self, data):
        func_step = getattr(self.model, "test_step", self.model.val_step)
        if callable(func_step):
            output = func_step(data)
        if output is not None:
            self.buffer.update_from_dict(output)


    def close(self):
        """close all registered hook before exit"""
        for _, hook in self._hook_dict.items():
            # Skip master_only hooks
            if hook is not None:
                hook.close()


    # ckpt related funcs
    def state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        for k in self.stat_dict_keys:
            state_dict[k] = getattr(self, k)
        state_dict['buffer'] = self.buffer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict):
        buffer_dict = state_dict.pop('buffer')
        self.buffer.load_state_dict(buffer_dict)
        for k, v in state_dict.items():
            setattr(self, k, v)


    # Misc
    def change_net_val(self):
        for _, net in self.model.nets.items():
            net.eval()

    def change_net_train(self):
        for _, net in self.model.nets.items():
            net.train()