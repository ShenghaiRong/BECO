from abc import abstractmethod

from torch.cuda.amp import GradScaler

from hooks.base_hook import BaseHook
from utils.registers import HOOKS_REG


class BaseOptimizerHook(BaseHook):
    @abstractmethod
    def backward(self, runner):
        pass


@HOOKS_REG.register_module()
class OptimizerHook(BaseOptimizerHook):
    def backward(self, runner):
        runner.buffer.get('loss').backward()
        runner.optimizer.step()
        runner.optimizer.zero_grad(set_to_none=True)


@HOOKS_REG.register_module()
class AMPOptimizerHook(BaseOptimizerHook):
    def __init__(self) -> None:
        self._loss_scaler = GradScaler()
    
    def backward(self, runner):
        self._loss_scaler.scale(runner.buffer.get('loss')).backward()
        self._loss_scaler.step(runner.optimizer)
        self._loss_scaler.update()
        runner.optimizer.zero_grad(set_to_none=True)

@HOOKS_REG.register_module()
class WeightingOptimizerHook(BaseOptimizerHook):
    def __init__(self) -> None:
        self.kwargs = {'mgda_gn': 'loss+'}

    def backward(self, runner):
        losses : list = runner.buffer.get('loss_list')
        w = runner.model.backward(losses, **self.kwargs)
        runner.buffer.update('loss_weight', w)
        runner.optimizer.step()
        runner.optimizer.zero_grad(set_to_none=True)


@HOOKS_REG.register_module()
class AMPWeightingOptimizerHook(WeightingOptimizerHook):
    def __init__(self) -> None:
        self._loss_scaler = GradScaler()
    
    def backward(self, runner):
        losses = self._loss_scaler.scale(runner.buffer.get('loss'))
        w = runner.model.backward(losses, **self.kwargs)
        runner.buffer.update('loss_weight', w)
        self._loss_scaler.step(runner.optimizer)
        self._loss_scaler.update()
        runner.optimizer.zero_grad(set_to_none=True)

@HOOKS_REG.register_module()
class GradientCumulativeOptimizerHook(BaseOptimizerHook):
    """
    multi-iters gradient cumulating hook, usually to simulate a large batch size

    Args:
        cumulative_iters (int): Num of gradient cumulative iters. The optimizer
        will step every `cumulative_iters` iters.

    NOTE: GradientCumulativeOptimizerHook may slightly decrease performance if the
    model has BatchNorm layers.
    """
    def __init__(self, runner, cumulative_iters:int=1) -> None:
        self.cumulative_iters = cumulative_iters
        # Integrity checking for resuming training
        if runner.iter % self.cumulative_iters != 0:
            runner.logger.warning(
                'Resume iter number is not divisible by cumulative_iters in '
                'GradientCumulativeOptimizerHook, which means the gradient of '
                'some iters is lost and the result may be influenced slightly.'
            )
        residual_iters = runner.max_iters - runner.iter
        self.divisible_iters = residual_iters // self.cumulative_iters * self.cumulative_iters
        self.remainder_iters = residual_iters - self.divisible_iters

    def backward(self, runner):
        # Last few iters should use different factor
        if runner.iter < self.divisible_iters:
            loss_factor = self.cumulative_iters
        else:
            loss_factor = self.remainder_iters
        loss = runner.buffer.get('loss')
        loss = loss / loss_factor
        loss.backward()

        # Update parameters every n iters
        if (self.every_n_iters(runner, self.cumulative_iters) or self.is_last_iter(runner)):
            runner.optimizer.step()
            runner.optimizer.zero_grad(set_to_none=True)
