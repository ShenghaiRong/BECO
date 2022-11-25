from typing import List, Dict

from torch.optim import lr_scheduler


def get_scheduler(sched_configs: List, optim):
    # Trying to find scheduler implemented by pytorch
    try:
        sched_cls = getattr(lr_scheduler, sched_configs['type'])
    # If it does not exist, find it in Register
    except Exception as e:
        try:
            #sched_cls = SCHEDULER_REG.get(sched_configs.type)
            sched_cls = PolyLR
        except Exception as e:
            raise RuntimeError(f"Error occured when trying to build scheduler{sched_configs['type']}")

    # If optimizer is not used, return None
    return sched_cls(optimizer = optim, **sched_configs['settings']) if optim else None


class PolyLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_power=0.9, min_lr=1e-5, last_epoch=-1):
        # self.max_iters should be updated later
        self.max_iters = 1e4
        self.power = lr_power
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch/self.max_iters)**self.power, self.min_lr)
                for base_lr in self.base_lrs]