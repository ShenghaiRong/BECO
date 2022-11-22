from abc import abstractmethod


from hooks.base_hook import BaseHook
from utils.registers import HOOKS_REG

@HOOKS_REG.register_module()
class CustomDatasetHook(BaseHook):
    """
    Model Custom pipeline hook. Default is pass, user should implement specific
    function in {model}.py
    """
    def __init__(self, keys) -> None:
        self.keys = keys

    def set_logging_path(self, runner):
        logging_path = runner.args.logging_path
        for key in self.keys:
            runner.dataloaders[key].dataset.logging_path = logging_path
            

