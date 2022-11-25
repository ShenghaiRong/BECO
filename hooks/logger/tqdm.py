from copy import deepcopy
from typing import List, Dict, Any
import time
import inspect

from tqdm import tqdm

from hooks.base_hook import BaseHook


class TqdmLoggerHook(BaseHook):
    """
    This is an implement for using tqdm and logging at the same. Messages to be
    logged to console should always use tqdm.write() to avoid breaking the progress bar
    
    This is designed to have two message bars, one for displaying train_metrics
    and the other for displaying val_metrics
    
    There will be two progress bars to display total progress and current workflow progess
    This class does not support customizing the numbers of bars

    Parameters:
        train_log_interval: The update interval for training metric bar

    About control sequence, more details can be found in 
    https://misc.flogisoft.com/bash/tip_colors_and_formatting
    """
    master_only = True
    
    def __init__(
            self,
            by_epoch: bool,
            train_log_interval: int = 10
        ) -> None:

        self.train_log_interval = train_log_interval
        self.by_epoch = by_epoch

        # bar_total and bar_iter is first initialized to take its position
        self.bar_total = tqdm(disable=True, leave=False, position=3)
        self.bar_iter = tqdm(disable=True, leave=False, position=2)
        self.bar_val = tqdm(total=1, bar_format='{desc}', leave=False, position=1)
        self.bar_train = tqdm(total=1, bar_format='{desc}', leave=False, position=0)

    def init_bar_total(self, runner):
        """Should be called on `before_run`"""
        if self.by_epoch:
            self._init_bar_total(initial=runner.epoch, total=runner.max_epoch)
        else:
            self._init_bar_total(initial=runner.iter, total=runner.max_iter)
        
    def _init_bar_total(self, initial: int=0, total: int=None) -> None:
        self.bar_total = tqdm(
            total=total,
            initial=initial,
            position=3,
            dynamic_ncols=False,
            ncols=100,
            leave=False
        )

    def init_bar_iter_train(self, runner):
        """Should be called on `before train`"""
        if self.by_epoch:
            self._init_bar_iter(total=
                runner._workflow['train'] * len(runner.dataloaders['train']))
        else:
            self._init_bar_iter(total=
                runner._workflow['train'])

    def init_bar_iter_trainval(self, runner):
        """Should be called on `before trainval`"""
        if self.by_epoch:
            self._init_bar_iter(total=
             runner._workflow['trainval'] * len(runner.dataloaders['trainval']))
        else:
            self._init_bar_iter(total=
                runner._workflow['trainval'])

    def init_bar_iter_val(self, runner):
        """Should be called on `before_val`"""
        if self.by_epoch:
            self._init_bar_iter(total=
                runner._workflow['val'] * len(runner.dataloaders['val']))
        else:
            self._init_bar_iter(total=
                runner._workflow['val'] if runner._workflow['val'] > -1
                else len(runner.dataloaders['val']))

    def init_bar_iter_test(self, runner):
        """Should be called on `before_test`"""
        self._init_bar_iter(total=len(runner.dataloaders['test']))

    def _init_bar_iter(self, total: int=None) -> None:
        self.bar_iter = tqdm(
            total=total,
            position=2,
            dynamic_ncols=False,
            ncols=100,
            leave=False
        )

    def update_bar_total(self, runner):
        self._update_bar_total()

    def _update_bar_total(self, step: int = 1):
        self.bar_total.update(step)

    def update_bar_iter(self, runner):
        self._update_bar_iter()

    def _update_bar_iter(self, step: int = 1):
        self.bar_iter.update(step)

    def log_train(self, runner):
        if self.every_n_iters(runner, self.train_log_interval):
            metrics = runner.buffer.get('train_metric')
            #metrics = remove_non_digit_metric(metrics)
            loss = runner.buffer.get('loss_dict')
            lr = runner.buffer.get('lr')
            self._log_train(lr, loss, metrics)

    def _log_train(self, lr: List, loss: Dict[str, float], 
                   metrics: Dict[str, float]):
        log_dict = dict()
        # Prevent changing original lr in runner.buffer
        lr = deepcopy(lr)
        for i, item in enumerate(lr):
            lr[i] = round(item, 5)
        log_dict['lr'] = str(lr)
        for k, v in loss.items():
            log_dict[k] = v
        if metrics is not None:
            for k, v in metrics.items():
                log_dict[k] = v
        msg = Msg.get_msg(5, log_dict)
        self.bar_train.set_description_str("Train_metrics: " + msg)

    def log_val(self, runner):
        """Should be called on `after_val_epoch`"""
        metrics = runner.buffer.get('val_metric')
        #metrics = remove_non_digit_metric(metrics)
        best_metric = runner.buffer.get('best_metric')
        metrics['Best'] = best_metric
        self._log_val(metrics)

    def _log_val(self, val_metrics: Dict[str, Any]):
        msg = Msg.get_msg(5, val_metrics)
        self.bar_val.set_description_str("Val_metrics: " + msg)

    def log_test(self, runner):
        metrics = runner.buffer.get('test_metric')
        #metrics = remove_non_digit_metric(metrics)
        msg = Msg.get_msg(5, metrics)
        self.write("Final test results: " + msg)

    def write(self, msg: str) -> None:
        msg = self._formate_str(msg)
        # Use the bar at top to output message
        self.bar_train.write(msg)

    def _formate_str(self, msg: str) -> str:
        """
        Generate similar output as logging.Logger
        """
        # getting the name of the module calling this function
        frame_records = inspect.stack()[2]
        mod_name = inspect.getmodulename(frame_records[1])
        cur_time = time.strftime("%H:%M:%S", time.localtime())
        return cur_time + " " + mod_name + " INFO " + msg

    def close(self):
        self.bar_train.close()
        self.bar_val.close()
        self.bar_iter.close()
        self.bar_total.close()


class Msg:

    @staticmethod
    def get_msg(digits_num: int, metric: Dict):
        str_list = list()
        for k, v in metric.items():
            if isinstance(v, float):
                v = round(v, digits_num)
            str_tmp = str(k) + ":" + str(v) + ", "
            str_list.append(str_tmp)

        msg_str = ""
        for item in str_list:
            msg_str += item

        return msg_str