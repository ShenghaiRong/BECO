import os
import shutil
import glob

import torch
from .base_hook import BaseHook

from utils.distributed import get_device


class CheckpointSaveHook(BaseHook):
    """
    Args:
        path: path to save ckpt
        by_epoch: whether interval is calculated by epoch
        best_metric_name: the name of metric for saving best ckpt
        max_ckpts: max number of ckpt to keep, 0 means unlimited
        del_ckpt: whether to delete all ckpts after training
    """
    master_only = True
    
    def __init__(
            self, path: str, by_epoch: bool, interval: int, best_metric_name: str,
            max_ckpts: int=1, del_ckpt: bool=False
        ) -> None:
        self.path = path
        self.by_epoch = by_epoch
        self.interval = interval
        self.best_metric_name = best_metric_name
        self.max_ckpts = max_ckpts
        self.del_ckpt = del_ckpt
        self.best_metric = 0.
        self.ckpt_name = 'epoch_{}.pth' if by_epoch else 'iter_{}.pth'
        self._init_best_record()

    def _interval_trigger(self, runner) -> bool:
        if self.interval > 1:
            if self.by_epoch:
                return self.every_n_epochs(runner, self.interval)
            else:
                return self.every_n_iters(runner, self.interval)
        else:
            return True

    def _get_filename(self, runner, step: int=None):
        if self.by_epoch:
            step = runner.epoch if step == None else step
            filename = os.path.join(self.path, f'ckpt_epoch_{step}.pth')
        else:
            step = runner.iter if step == None else step
            filename = os.path.join(self.path, f'ckpt_iter_{step}.pth')
        return filename

    def _remove_old_ckpt(self, runner):
        current = runner.epoch if self.by_epoch else runner.iter

        if self.max_ckpts > 0:
            old_ckpt_steps = range(current - self.max_ckpts * self.interval, 0, -self.interval)
            for _step in old_ckpt_steps:
                old_ckpt_path = self._get_filename(runner, _step)
                if os.path.exists(old_ckpt_path):
                    os.remove(old_ckpt_path)

    def _init_best_record(self):
        save_best = os.path.join(self.path, f'best={100*self.best_metric:.2f}')
        best_dir = glob.glob(os.path.join(self.path, 'best=*'))
        if len(best_dir) > 0:
            for idx in range(len(best_dir)):
                os.rmdir(best_dir[idx])
        os.mkdir(save_best)

    def get_net_state_dict(self, runner):
        net_state_dict = dict()
        for k, v in runner.net_dict.items():
            net_state_dict[k] = v.state_dict()
        return net_state_dict

    def get_optim_state_dict(self, runner):
        return runner.optimizer.state_dict()

    def get_sched_state_dict(self, runner):
        return runner.scheduler.state_dict()

    def get_model_state_dict(self, runner):
        return runner.model.state_dict()

    def get_runner_state_dict(self, runner):
        return runner.state_dict()

    def save_ckpt(self, runner):
        """
        Save a general checkpoint that can resume training
        """
        if self._interval_trigger(runner):
            runner.logger.write("Saving latest ckpt...")
            ckpt = dict(
                net_state_dict = self.get_net_state_dict(runner),
                optim_state_dict = self.get_optim_state_dict(runner),
                sched_state_dict = self.get_sched_state_dict(runner),
                model_state_dict = self.get_model_state_dict(runner),
                runner_state_dict = self.get_runner_state_dict(runner),
            )
            filename = self._get_filename(runner)
            torch.save(ckpt, filename)
            self._remove_old_ckpt(runner)

    def save_model(self, runner):
        """
        Save only the network parameters
        """
        ckpt = dict(
            net_state_dict = self.get_net_state_dict(runner),
        )
        torch.save(ckpt, os.path.join(self.path, "best_ckpt.pth"))

    def save_best_model(self, runner):
        # best_metric should be a float type
        current_val_metric = runner.buffer.get('val_metric')[self.best_metric_name]
        # update best metric and save ckpt
        if current_val_metric > self.best_metric:
            old_best = f'best={100*self.best_metric:.2f}'
            new_best = f'best={100*current_val_metric:.2f}'
            os.rename(os.path.join(self.path, old_best),
                      os.path.join(self.path, new_best))
            self.best_metric = current_val_metric
            runner.logger.write("Saving best ckpt...")
            ckpt = dict(net_state_dict = self.get_net_state_dict(runner))
            torch.save(ckpt, os.path.join(self.path, "best_ckpt.pth"))

    def close(self):
        """Remove all old ckpts to save disk space"""
        if self.del_ckpt:
            pattern = 'ckpt_epoch_' if self.by_epoch else 'ckpt_iter_'
            files = [f for f in os.listdir(self.path) if pattern in f]
            for f in files:
                os.remove(os.path.join(self.path, f))


class CheckpointLoadHook(BaseHook):
    """
    NOTE: This hook is used to load the checkpoint given when initialized. If you want
    to dynamicly load ckpt several times during training, you should implement a
    new hook yourself

    The network in runner but not presented in ckpt will not be touched
    The network in ckpt but not presented in runner will not be used
    """
    def __init__(self, path:str, net_only: bool=False, strict: bool=True) -> None:
        self.path = path
        self.net_only = net_only
        self.strict = strict
    
    def load_ckpt(self, runner):
        """
        Load a general checkpoint to resume training
        """
        if self.path is None:
            return
        
        runner.logger.write("Loading ckpt...")
        device = get_device()
        ckpt = torch.load(self.path, map_location=device)
        
        if 'net_state_dict' in ckpt:
            runner.logger.write("Loading net state...")
            self.load_net_state_dict(runner, ckpt['net_state_dict'])
        if self.net_only:
            return

        if 'runner_state_dict' in ckpt:
            runner.logger.write("Loading runner state...")
            self.load_runner_state_dict(runner, ckpt['runner_state_dict'])
        if 'optim_state_dict' in ckpt:
            runner.logger.write("Loading optim state...")
            self.load_optim_state_dict(runner, ckpt['optim_state_dict'])
        if 'sched_state_dict' in ckpt:
            runner.logger.write("Loading scheduler state...")
            self.load_sched_state_dict(runner, ckpt['sched_state_dict'])
        if 'model_state_dict' in ckpt:
            runner.logger.write("Loading model state...")
            self.load_model_state_dict(runner, ckpt['model_state_dict'])
        
    def load_net_state_dict(self, runner, state_dict):
        for k, v in state_dict.items():
            if k in runner.net_dict:
                runner.net_dict[k].load_state_dict(v, strict = self.strict)

    def load_optim_state_dict(self, runner, state_dict):
        runner.optimizer.load_state_dict(state_dict)

    def load_sched_state_dict(self, runner, state_dict):
        runner.scheduler.load_state_dict(state_dict)

    def load_model_state_dict(self, runner, state_dict):
        runner.model.load_state_dict(state_dict)

    def load_runner_state_dict(self, runner, state_dict):
        runner.load_state_dict(state_dict)


class RestoreCkptHook(BaseHook):
    """
    Restore best ckpt before validation
    """
    def __init__(self, path: str) -> None:
        self.best_model = os.path.join(path, "best_ckpt.pth")

    def load_ckpt(self, runner):
        runner.logger.write("Restore to best ckpt...")
        device = get_device()
        ckpt = torch.load(self.best_model, map_location=device)
        ckpt = ckpt['net_state_dict']
        for k, v in runner.net_dict.items():
            v.load_state_dict(ckpt[k])


