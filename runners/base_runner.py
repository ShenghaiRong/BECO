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
        config: The FULL config object
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
        config,
        model,
        optimizer,
        scheduler,
        dataloaders: Dict,
        samplers: Dict,
        workflow: Dict,
        **kwargs
    ) -> None:
        
        self.args = args
        self.config = config
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

        self.register_hook(config.hooks)

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
        self.call_hook('before_test')
        for i, data in enumerate(self.dataloaders['test']):
            self.call_hook('before_test_iter')
            self.test_iter(data)
            self.inner_iter = i
            self.call_hook('after_test_iter')
        self.call_hook('after_test')

    
    def test_iter(self, data):
        func_step = getattr(self.model, "test_step", self.model.val_step)
        if callable(func_step):
            output = func_step(data)
        if output is not None:
            self.buffer.update_from_dict(output)


    # hook related funcs
    def _init_hook(self):
        """
        self._hooks is a dict of arraies, with each item holding list of hooks
        of each stages. All supported stages is defined here
        """
        self._hooks = dict()
        for k in hooks.BaseHook.stages:
            self._hooks[k] = list()

    def register_hook(self, hooks_cfg: List):
        """
        register hooks at the sequence given in config file
        """
        self._init_hook()
        self._hook_dict = hooks.get_hooks(self.config.hooks)
        for stage in self._hooks.keys():
            for item in hooks_cfg.stages[stage]:
                hook_name, hook_func = item.split('.')
                # process model based hooks
                if hook_name == 'model':
                    self._hooks[stage].append(getattr(self.model, hook_func))
                # skip master_only hooks
                elif self._hook_dict[hook_name] is None:
                    continue
                else:
                    self._hooks[stage].append(getattr(self._hook_dict[hook_name], hook_func))

    def get_hook_info(self) -> Dict:
        """
        return a dict containing all hooks name of each stage
        """
        stage_hook_map = {stage: [] for stage in self._hooks.keys()}
        for stage, hooks in self._hooks.items():
            for hook in hooks:
                stage_hook_map[stage].append(hook.__name__)
        return stage_hook_map

    def call_hook(self, stage: str):
        for hook in self._hooks[stage]:
            hook(self)

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