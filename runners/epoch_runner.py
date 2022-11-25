import time

import torch

from .base_runner import BaseRunner
from utils.logger import DummyLogger


class EpochRunner(BaseRunner):
    def _init(self, max_epoch):
        self.max_epoch = max_epoch
        # Register TqdmLoggerHook as default logger
        if self._hook_dict['TqdmLoggerHook'] is not None:
            self.logger = self._hook_dict['TqdmLoggerHook']
        else:
            self.logger = DummyLogger()

    def _set_epoch(self):
        """set epoch for distributed data sampler"""
        if hasattr(self.dataloaders['train'].sampler, 'set_epoch'):
            self.dataloaders['train'].sampler.set_epoch(self.epoch)
    
    def run_iter(self, data):
        """Run for one iteration"""
        if self.is_train:
            output = self.model.train_step(data)
        else:
            output = self.model.val_step(data)
        # The output of model should be a dict
        if output is not None:
            self.buffer.update_from_dict(output)

    def train(self):
        """Train on train set for one epoch"""
        self.change_net_train()
        self._set_epoch()

        #self.call_hook("before_train_epoch")
        self._hook_dict['CustomModelHook'].before_train_epoch(self)

        for i, data in enumerate(self.dataloaders['train']):
            #self.call_hook('before_train_iter')
            self._hook_dict['CustomModelHook'].before_train_iter(self)

            self.run_iter(data)
            self.inner_iter = i
            self.iter += 1
            #self.call_hook('after_train_iter')
            self._hook_dict['OptimizerHook'].backward(self)
            self._hook_dict['SchedulerHook'].step(self)
            self._hook_dict['CustomModelHook'].after_train_iter(self)
            self._hook_dict['MetricHook'].get_lr(self)
            self._hook_dict['MetricHook'].get_train_loss(self)
            if self._hook_dict['TqdmLoggerHook'] is not None:
                self._hook_dict['TqdmLoggerHook'].log_train(self)
                self._hook_dict['TqdmLoggerHook'].update_bar_iter(self)
            if self._hook_dict['TBLoggerHook'] is not None:
                self._hook_dict['TBLoggerHook'].log_train(self)

        self.epoch += 1

        #self.call_hook('after_train_epoch')
        if self._hook_dict['TqdmLoggerHook'] is not None:
            self._hook_dict['TqdmLoggerHook'].update_bar_total(self)
        self._hook_dict['CustomModelHook'].after_train_epoch(self)
        if self._hook_dict['CheckpointSaveHook'] is not None:
            self._hook_dict['CheckpointSaveHook'].save_ckpt(self)

    @torch.no_grad()
    def val(self):
        """Val on val set for one epoch"""
        self.change_net_val()
        for i, data in enumerate(self.dataloaders['val']):
            #self.call_hook('before_val_iter')

            self.run_iter(data)
            self.inner_iter = i
            #self.call_hook('after_val_iter')
            if self._hook_dict['TqdmLoggerHook'] is not None:
                self._hook_dict['TqdmLoggerHook'].update_bar_iter(self)

    def run(self):
        """Start training"""
        #self.call_hook('before_run')
        self._hook_dict['CheckpointLoadHook'].load_ckpt(self)
        self._hook_dict['PolyLRInitHook'].init_polylr(self)
        if self._hook_dict['TqdmLoggerHook'] is not None:
            self._hook_dict['TqdmLoggerHook'].init_bar_total(self)
        self._hook_dict['CustomModelHook'].before_run(self)

        self.logger.write("Start training...")
        while self.epoch < self.max_epoch:
            for mode, epochs in self._workflow.items():
                assert mode in ('train', 'val')
                # get train of val function
                epoch_runner = getattr(self, mode)
                self._prepare_workflow(mode)        
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self.max_epoch:
                        break
                    epoch_runner()
                self._end_workflow(mode)
        self.logger.write("Finished training")
        #self.call_hook('after_run')
        self._hook_dict['CustomModelHook'].after_run(self)
    
    def _prepare_workflow(self, mode):
        if mode == 'train':
            self.logger.write(f"Start training of epoch {self.epoch}")
            self.is_train = True
            self.change_net_train()
            #self.call_hook('before_train')
            if self._hook_dict['TqdmLoggerHook'] is not None:
                self._hook_dict['TqdmLoggerHook'].init_bar_iter_train(self)
        else:
            self.logger.write(f"Start validating...")
            self.is_train = False
            self.change_net_val()
            #self.call_hook('before_val')
            if self._hook_dict['TqdmLoggerHook'] is not None:
                self._hook_dict['TqdmLoggerHook'].init_bar_iter_val(self)
            self._hook_dict['MetricHook'].reset_metric(self)
            self._hook_dict['CustomModelHook'].before_val(self)
        self.start_time = time.time()

    def _end_workflow(self, mode):
        if mode == 'train':
            self.logger.write(f"Time taken for training on epoch {self.epoch - 1}"
                f" : {round((time.time() - self.start_time) / 60, 2)} mins")
            #self.call_hook('after_train')
        else:
            self.logger.write(f"Time taken for validating"
                f" : {round((time.time() - self.start_time) / 60, 2)} mins")
            #self.call_hook('after_val')
            self._hook_dict['MetricHook'].get_val_metric(self)
            if self._hook_dict['TqdmLoggerHook'] is not None:
                self._hook_dict['TqdmLoggerHook'].log_val(self)
            if self._hook_dict['TBLoggerHook'] is not None:
                self._hook_dict['TBLoggerHook'].log_val(self)
            if self._hook_dict['CheckpointSaveHook'] is not None:
                self._hook_dict['CheckpointSaveHook'].save_best_model(self)
            self._hook_dict['CustomModelHook'].after_val(self)