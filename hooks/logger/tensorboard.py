from torch.utils.tensorboard import SummaryWriter
import torch

from hooks.base_hook import BaseHook
from utils.misc import derivation
from utils.registers import HOOKS_REG
from utils.visual.plot import plot_confusion_matrix


@HOOKS_REG.register_module()
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
        # Stage is only available in incremental learning setups
        stage = runner.config.datasets.settings.step
        step = runner.iter
        metrics_dict = runner.buffer.get('val_metric')

        # log val metrics
        self.writer.add_scalars(self.tag_table['val'], metrics_dict, step)

    def log_test(self, runner):
        metrics_dict = runner.buffer.get('test_metric')
        self.writer.add_scalars("Test", metrics_dict)
    
    def close(self):
        self.writer.flush()
        self.writer.close()



@HOOKS_REG.register_module()
class CustomTBLoggerHook(TBLoggerHook):
    def __init__(self, path: str, log_interval: int = 50, 
                 custom_key: str = None) -> None:
        super().__init__(path, log_interval)
        self.custom_key = custom_key 

    def log_multi_metrics(self, runner):
        if self.every_n_iters(runner, self.log_interval):
            step = runner.iter
            metrics_dict = runner.buffer.get(self.custom_key)
            classes_map = runner.dataloaders['val'].dataset.classes_map
            for i, cls in enumerate(classes_map):
                cls_iou = {}
                for key, dict in metrics_dict.items():
                    iou_list = dict['IoU']
                    iou = round(iou_list[i] * 100, 1)
                    cls_iou[key] = iou
                self.writer.add_scalars(f"Class_IoU/{cls}", cls_iou, step)

    def log_multi_metrics_epoch(self, runner):
        epoch = runner.epoch
        metrics_dict = runner.buffer.get(self.custom_key)
        classes_map = runner.dataloaders['val'].dataset.classes_map
        for i, cls in enumerate(classes_map):
            cls_iou = {}
            cls_acc = {}
            for key, dict in metrics_dict.items():
                iou_list = dict['IoU']
                acc_list = dict['Class_Acc']
                iou = round(iou_list[i] * 100, 1)
                acc = round(acc_list[i] * 100, 1)
                cls_iou[key] = iou
                cls_acc[key] = acc
            self.writer.add_scalars(f"Train/{cls}", cls_iou, epoch)
            ps_cls_dict = {'ps_iou': cls_iou['pseu_labels'], 
                            'ps_acc': cls_acc['pseu_labels']}
            self.writer.add_scalars(f"AnaDY/{cls}", ps_cls_dict, epoch)

    def log_updated_class(self, runner):
        cur_updated_class = runner.model.get_updated_class()
        classes_map = runner.dataloaders['val'].dataset.classes_map
        updated_class_dict = {}
        for cls_idx in cur_updated_class:
            updated_class_dict[classes_map[cls_idx]] = cls_idx
            self.writer.add_scalars('Updated_Class', updated_class_dict, 
                                    runner.epoch)

    def log_correction_count(self, runner):
        #only use in after_trainval
        epoch = runner.epoch
        count, theta = runner.model.get_correction_count()
        self.writer.add_scalar('correction_count', count, epoch)
        self.writer.add_scalar('correction_theta', theta, epoch)
        count_list, theta_list = runner.model.get_correction_count_list()
        classes_map = runner.dataloaders['val'].dataset.classes_map
        for i, cls in enumerate(classes_map):
            cls_count = count_list[i]
            cls_theta = theta_list[i]
            cls_count_d10k = round(cls_count / 10000, 2)
            self.writer.add_scalar(f"cls_correct_count/{cls}", cls_count, epoch)
            self.writer.add_scalars(f"AnaDY/{cls}", 
                                    {'count_div10k': cls_count_d10k}, epoch)
            self.writer.add_scalar(f"cls_theta/{cls}", cls_theta, epoch)

    def log_dylabel_metric(self, runner):
        #only use in after_trainval
        epoch = runner.epoch
        metrics_dict = runner.buffer.get('dylabel_metric')
        classes_map = runner.dataloaders['val'].dataset.classes_map
        if epoch == 1:
            runner.buffer.update('ori_dylabel_metric', metrics_dict)
        else:
            ori_metrics_dict = runner.buffer.get('ori_dylabel_metric')
            ori_iou = ori_metrics_dict['IoU']
            ori_miou = ori_metrics_dict['mIoU']
            ori_acc = ori_metrics_dict['Class_Acc']
            ori_macc = ori_metrics_dict['mClass_Acc']
            current_iou = metrics_dict['IoU']
            current_miou = metrics_dict['mIoU']
            current_acc = metrics_dict['Class_Acc']
            current_macc = metrics_dict['mClass_Acc']
            delta = current_miou - ori_miou
            delta_acc = current_macc - ori_macc
            delta = round(delta * 100, 2)
            delta_acc = round(delta_acc * 100, 2)
            delta_dict = {'dylabel_delta': delta, 'delta_acc': delta_acc}
            self.writer.add_scalars("AnaDY/z-mean", delta_dict, epoch)
            for i, cls in enumerate(classes_map):
                delta = current_iou[i] - ori_iou[i]
                delta = round(delta * 100, 2)
                delta_acc = current_acc[i] - ori_acc[i]
                delta_acc = round(delta_acc * 100, 2)
                delta_dict = {'dylabel_delta': delta, 'delta_acc': delta_acc}
                self.writer.add_scalars(f"AnaDY/{cls}", delta_dict, epoch)

    def log_logit_norm(self, runner):
        epoch = runner.epoch
        logit_norm = runner.model.get_logit_norm()
        logit_norm_dict = {'logit_norm': logit_norm}
        self.writer.add_scalars("AnaDY/z-mean", logit_norm_dict, epoch)

    def log_cls_derivation(self, runner):
        epoch = runner.epoch
        cls_der_list = runner.model.get_cls_derivation()
        classes_map = runner.dataloaders['val'].dataset.classes_map
        for i, cls in enumerate(classes_map):
            derivat = round(cls_der_list[i], 2)
            der_dict = {'derivation%': derivat}
            self.writer.add_scalars(f"AnaDY/{cls}", der_dict, epoch)

    def log_confusion_matrix(self, runner):
        matrix: torch.Tensor = runner.buffer.get('matrix').cpu()
        epoch = runner.epoch

        plot = plot_confusion_matrix(matrix)
        self.writer.add_figure('Fig_CMatrix', plot, epoch)

    def log_loss_weight(self, runner):
        w_list: torch.Tensor = runner.buffer.get('loss_weight')
        step = runner.iter

        for i in range(w_list.size(0)):
            w_dict = {f"loss_{i}": w_list[i]}
            self.writer.add_scalars("loss_weight", w_dict, step)





