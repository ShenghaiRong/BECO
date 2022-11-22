from typing import Dict, List

from .base_hook import BaseHook
from utils.registers import HOOKS_REG
from utils.distributed import is_distributed, get_dist_info


@HOOKS_REG.register_module()
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


@HOOKS_REG.register_module()
class CustomMetricHook(BaseHook):
    """
    Customize the metric format to get
    """
    def __init__(self, metric_keys: List[str], buffer_key: str) -> None:
        self.metric_keys = metric_keys
        self.buffer_key = buffer_key

    def get_metric(self, runner):
        metric_dict = runner.model.metric.get_results(self.metric_keys)
        runner.buffer.update(self.buffer_key, metric_dict)


    def __init__(self, interval: int, metric_keys: List[str], 
                 best_metric_name: str = None, custom_key: str='ck') -> None:
        super().__init__(interval, metric_keys, best_metric_name)
        self.custom_key = custom_key

    def get_val_metric(self, runner):
        cur_metric_dict = self._get_ana_metric(runner)
        cur_metric = cur_metric_dict['gt_labels'].copy()
        self.reset_ana_metric(runner)
        _ = cur_metric.pop('IoU')
        _ = cur_metric.pop('Class_Acc')
        runner.buffer.update('val_metric', cur_metric)
        runner.buffer.update(self.custom_key, cur_metric_dict)
        #runner.model.update_iou(cur_metric_dict)
        # Update best metric value
        #if self.best_metric_name is not None:
        #    pre_best_metric = runner.buffer.get('best_metric')
        #    pre_best_metric = pre_best_metric if pre_best_metric else 0.
        #    cur_best_metric = cur_metric[self.best_metric_name]
        #    if cur_best_metric > pre_best_metric:
        #        runner.buffer.update('best_metric', cur_best_metric)

    def _get_ana_metric(self, runner) -> Dict:
        if is_distributed():
            runner.model.ana_metric.all_reduce()
        metric_dict = runner.model.ana_metric.get_results(self.metric_keys)
        return metric_dict

    def reset_ana_metric(self, runner):
        runner.model.ana_metric.reset()

    def get_train_metric(self, runner):
        if self.before_every_n_iters(runner, self.interval):
            # Let model prepare metric
            runner.model.metric_update_trigger = True
        if self.every_n_iters(runner, self.interval):
            #runner.model.metric_update_trigger = False
            runner.model.metric_update_trigger = True
            cur_metric_dict = self._get_ana_metric(runner)
            cur_metric = cur_metric_dict['pseu_labels'].copy()
            self.reset_ana_metric(runner)
            iou_list = cur_metric.pop('IoU')
            acc_list = cur_metric.pop('Class_Acc')
            if self.every_n_iters(runner, 5 * self.interval):
                self.reset_metric(runner)
                runner.model.update_iou_list(iou_list)
            runner.buffer.update('train_metric', cur_metric)
            runner.buffer.update(self.custom_key, cur_metric_dict)
            runner.buffer.update('loss_dict', runner.model.get_loss())

    def get_dylabel_metric(self, runner):
        metric = self._get_metric(runner)
        matrix = runner.model.metric.get_hist()
        self.reset_metric(runner)
        runner.buffer.update('dylabel_metric', metric)
        runner.buffer.update('matrix', matrix)
        
