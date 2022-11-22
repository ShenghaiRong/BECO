import time

import torch

from .base_runner import BaseRunner
from utils.registers import RUNNERS_REG
from utils.logger import DummyLogger


class IterLoader:

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            # for DDP training
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            # Prevent possible deadlock during epoch transition
            time.sleep(1)
            # re-init dataloader
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


@RUNNERS_REG.register_module()
class IterRunner(BaseRunner):
    def _init(self, max_iter):
        self.max_iter = max_iter
        # change all dataloaders to iterloaders
        self.iter_loaders = {k: IterLoader(v) for k, v in self.dataloaders.items()}
        if self._hook_dict['TqdmLoggerHook'] is not None:
            self.logger = self._hook_dict['TqdmLoggerHook']
        else:
            self.logger = DummyLogger()

    def run_iter(self, data):
        if self.is_train:
            output = self.model.train_step(data)
        else:
            output = self.model.val_step(data)
        # The output of model should be a dict
        if output is not None:
            self.buffer.update_from_dict(output)

    def train(self):
        """Train for one iteration"""
        self.call_hook('before_train_iter')
        data = next(self.iter_loaders['train'])
        self.run_iter(data)
        self.inner_iter += 1
        self.iter += 1
        self.call_hook('after_train_iter')

    @torch.no_grad()
    def val(self):
        """Val for one iteration"""
        self.call_hook('before_val_iter')
        data = next(self.iter_loaders['val'])
        self.run_iter(data)
        self.call_hook('after_val_iter')

    def run(self):
        """Start training"""
        self.call_hook('before_run')
        self.logger.write("Start Training...")
        while self.iter < self.max_iter:
            for mode, iters in self._workflow.items():
                assert mode in ('train', 'val')
                iter_runner = getattr(self, mode)
                self._prepare_workflow(mode)
                # -1 equals to run an epoch
                if iters == -1:
                    iters = len(self.dataloaders[mode])

                for _ in range(iters):
                    if mode == 'train' and self.iter >= self.max_iter:
                        break
                    iter_runner()
                
                self._end_workflow(mode)

        self.call_hook('after_run')


    def _prepare_workflow(self, mode):
        if mode == 'train':
            self.logger.write(f"Start training...")
            self.is_train = True
            self.change_net_train()
            self.call_hook('before_train')
        else:
            self.logger.write(f"Start validating...")
            self.is_train = False
            self.change_net_val()
            self.call_hook('before_val')

    def _end_workflow(self, mode):
        if mode == 'train':
            self.call_hook('after_train')
        else:
            self.call_hook('after_val')