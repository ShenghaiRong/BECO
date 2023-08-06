from torch.utils.tensorboard import SummaryWriter
import torch

from hooks.base_hook import BaseHook


class TBLoggerHook(BaseHook):
    """
    Tensorboard logger hook

    NOTE: The user might need to re-writer functions to suits the usage of different tasks

    Parameters:
        path: the logging path
        log_interval: the log interval for training
    """
    master_only = True

    def __init__(self, path: str, log_interval: int = 50) -> None:
        # Put logging contents in a subfolder for better organization
        path += "/tensorboard"
        self.writer = SummaryWriter(path, flush_secs=60)
        self.log_interval = log_interval
        # Predefined tags
        self.tag_table = dict(
            lr = 'LR',
            loss = 'Loss',
            train = 'Train',
            val = 'Val',
        )

    def get_writer(self, runner):
        runner.buffer.update('writer', self.writer)

    def log_train(self, runner):
        if self.every_n_iters(runner, self.log_interval):
            step = runner.iter
            loss_dict = runner.buffer.get('loss_dict')
            metrics_dict = runner.buffer.get('train_metric')
            lr = runner.buffer.get('lr')
            # change lr_list to lr_dict
            lr_dict = dict()
            for i, item in enumerate(lr):
                lr_dict[str(i)] = item

            self.writer.add_scalars(self.tag_table['lr'], lr_dict, step)
            # log training loss
            self.writer.add_scalars(self.tag_table['loss'], loss_dict, step)
            # loss training metrics
            if metrics_dict is not None:
                self.writer.add_scalars(self.tag_table['train'], metrics_dict, 
                                        step)

    def log_val(self, runner):

        step = runner.iter
        metrics_dict = runner.buffer.get('val_metric')

        # log val metrics
        if metrics_dict['Best'] is not None:
            self.writer.add_scalars(self.tag_table['val'], metrics_dict, step)

    def log_test(self, runner):
        metrics_dict = runner.buffer.get('test_metric')
        self.writer.add_scalars("Test", metrics_dict)
    
    def close(self):
        self.writer.flush()
        self.writer.close()






