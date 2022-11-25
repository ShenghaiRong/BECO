from typing import Dict

from . import checkpoint
from . import logger
from . import optimizer
from . import scheduler
from . import metric
from . import model_custom

from utils.distributed import get_dist_info

def get_hooks(logging_path, ckpt, amp) -> Dict:
    """General build function for hooks"""
    hook_dict = dict()

    hook_dict['CheckpointSaveHook'] = checkpoint.CheckpointSaveHook(
        path=logging_path,
        by_epoch=True,
        interval=1,
        best_metric_name="mIoU",
        max_ckpts=1,
        del_ckpt=True
    ) if get_dist_info()[0] == 0 else None

    hook_dict['CheckpointLoadHook'] = checkpoint.CheckpointLoadHook(
        path=ckpt,
        net_only=False
    )
    hook_dict['RestoreCkptHook'] = checkpoint.RestoreCkptHook(
        path=logging_path
    )
    hook_dict['TqdmLoggerHook'] = logger.tqdm.TqdmLoggerHook(
        by_epoch=True,
        train_log_interval=10,
    ) if get_dist_info()[0] == 0 else None

    hook_dict['TBLoggerHook'] = logger.tensorboard.TBLoggerHook(
        path=logging_path,
        log_interval=10
    ) if get_dist_info()[0] == 0 else None
    
    hook_dict['MetricHook'] = metric.MetricHook(
        interval=10,
        metric_keys=["Overall_Acc", "mClass_Acc", "mPred_Acc", "mIoU"],
        best_metric_name="mIoU"
    )
    if amp:
        hook_dict['OptimizerHook'] = optimizer.AMPOptimizerHook()
    else:
        hook_dict['OptimizerHook'] = optimizer.OptimizerHook()

    hook_dict['SchedulerHook'] = scheduler.SchedulerHook()

    hook_dict['PolyLRInitHook'] = scheduler.PolyLRInitHook()

    hook_dict['CustomModelHook'] = model_custom.CustomModelHook()

    return hook_dict