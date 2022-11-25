from abc import abstractmethod

from torch.cuda.amp import GradScaler

from hooks.base_hook import BaseHook


class BaseOptimizerHook(BaseHook):
    @abstractmethod
    def backward(self, runner):
        pass


class OptimizerHook(BaseOptimizerHook):
    def backward(self, runner):
        runner.buffer.get('loss').backward()
        runner.optimizer.step()
        runner.optimizer.zero_grad(set_to_none=True)


class AMPOptimizerHook(BaseOptimizerHook):
    def __init__(self) -> None:
        self._loss_scaler = GradScaler()
    
    def backward(self, runner):
        self._loss_scaler.scale(runner.buffer.get('loss')).backward()
        self._loss_scaler.step(runner.optimizer)
        self._loss_scaler.update()
        runner.optimizer.zero_grad(set_to_none=True)

