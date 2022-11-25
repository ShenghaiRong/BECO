from typing import Dict, List

from .base_hook import BaseHook
from utils.distributed import is_distributed, get_dist_info


class MetricHook(BaseHook):
    """
    Get metrics from model and put into runner.buffer. To get customized metrics,
        use `CustomMetricHook`.

    NOTE: This hook will reset model.metric after calling. This is a must
        if user want to implement their own hook
    
    Args:
        interval: interval to retrieve metric during training, defined as iterations.
        metric_keys: The metric name to be retrieved, refer to the Metric Class
            for its supported keys
        best_metric_name: Thed metric to be used as best metric
    """
    def __init__(self, interval: int, metric_keys: List[str], best_metric_name: str=None) -> None:
        self.interval = interval
        self.metric_keys = metric_keys
        self.best_metric_name = best_metric_name

    def get_lr(self, runner):
        """Return a list of learning rate of each param group"""
        if self.every_n_iters(runner, self.interval):
            lr_list =  [group['lr'] for group in runner.optimizer.param_groups]
            runner.buffer.update('lr', lr_list)

    def get_train_metric(self, runner):
        if self.before_every_n_iters(runner, self.interval):
            # Let model prepare metric
            runner.model.metric_update_trigger = True
        if self.every_n_iters(runner, self.interval):
            runner.model.metric_update_trigger = False
            metric = self._get_metric(runner)
            self.reset_metric(runner)
            runner.buffer.update('train_metric', metric)
            runner.buffer.update('loss_dict', runner.model.get_loss())
    
    def get_train_loss(self, runner):
        """Do not use this func if already use get_train_metric """
        if self.every_n_iters(runner, self.interval):
            runner.buffer.update('loss_dict', runner.model.get_loss())


    def get_val_metric(self, runner):
        cur_metric = self._get_metric(runner)
        self.reset_metric(runner)
        runner.buffer.update('val_metric', cur_metric)
        # Update best metric value
        if self.best_metric_name is not None:
            pre_best_metric = runner.buffer.get('best_metric')
            pre_best_metric = pre_best_metric if pre_best_metric else 0.
            cur_best_metric = cur_metric[self.best_metric_name]
            if cur_best_metric > pre_best_metric:
                runner.buffer.update('best_metric', cur_best_metric)

    def get_test_metric(self, runner):
        metric = self._get_metric(runner)
        self.reset_metric(runner)
        runner.buffer.update('test_metric', metric)

    def _get_metric(self, runner) -> Dict:
        if is_distributed():
            runner.model.metric.all_reduce()
        metric_dict = runner.model.metric.get_results(self.metric_keys)
        return metric_dict

    def reset_metric(self, runner):
        runner.model.metric.reset()
