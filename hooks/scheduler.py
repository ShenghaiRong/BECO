from hooks.base_hook import BaseHook
from utils.registers import HOOKS_REG


@HOOKS_REG.register_module()
class SchedulerHook(BaseHook):
    def __init__(self) -> None:
        pass

    def step(self, runner) -> None:
        runner.scheduler.step()
            

@HOOKS_REG.register_module()
class PolyLRInitHook(BaseHook):
    """
    This is used for initialzed PolyLR.max_iters after runner is built
    """
    def init_polylr(self, runner):
        by_epoch = hasattr(runner, 'max_epoch')
        runner.scheduler.max_iters = \
            runner.max_epoch * len(runner.dataloaders['train']) if by_epoch else runner.max_iter
        # Make resume training working properly
        runner.scheduler.last_epoch = runner.iter