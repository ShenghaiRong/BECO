from abc import abstractmethod


from hooks.base_hook import BaseHook

class CustomModelHook(BaseHook):
    """
    Model Custom pipeline hook. Default is pass, user should implement specific
    function in {model}.py
    """
    def before_run(self, runner):
        runner.model.before_run()

    def after_run(self, runner):
        runner.model.after_run()

    def before_train_epoch(self, runner):
        runner.model.before_train_epoch()

    def after_train_epoch(self, runner):
        runner.model.after_train_epoch()

    def before_train_iter(self, runner):
        runner.model.before_train_iter()

    def after_train_iter(self, runner):
        runner.model.after_train_iter()

    def before_val(self, runner):
        runner.model.before_val()

    def after_val(self, runner):
        runner.model.after_val()

    def before_trainval(self, runner):
        runner.model.before_trainval()

    def after_trainval(self, runner):
        runner.model.after_trainval()

    def get_logging_path(self, runner):
        logging_path = runner.args.logging_path
        runner.model.get_logging_path(logging_path)

