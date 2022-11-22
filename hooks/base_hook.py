class BaseHook:
    """
    Cuz that validation / test is usually done on full val/test datasets once. 
    So there is no `val_epoch` or `test_epoch` related stage
    """

    # This array lists supported stages for hooks. If want to add more stages, 
    # This should be updated accordingly with `Baserunner`
    stages = [
        'before_run', 'after_run',
        'before_train', 'after_train', 
        'before_train_epoch', 'after_train_epoch',
        'before_train_iter', 'after_train_iter',
        'before_val', 'after_val',
        'before_val_iter', 'after_val_iter',
        'before_test', 'after_test',
        'before_test_iter', 'after_test_iter',
        'before_trainval', 'after_trainval',
        'before_trainval_iter', 'after_trainval_iter',
    ]

    # This array lists all supported trigger conditions for hooks. A new 
    # trigger condition is defined by a function returning a bool value
    conditions = [
        'every_n_epochs', 'every_n_inner_iters', 'every_n_iters',
        'is_last_epoch', 'is_last_iter'
    ]

    # Every trigger condition has a "before" variety to trigger operations
    # one iter / epoch earlier for the need of preparation in advance

    def every_n_epochs(self, runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def before_every_n_epochs(self, runner, n):
        return (runner.epoch + 2) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n):
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def before_every_n_inner_iters(self, runner, n):
        return (runner.inner_iter + 2) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def before_every_n_iters(self, runner, n):
        return (runner.iter + 2) % n == 0 if n > 0 else False

    def is_last_epoch(self, runner):
        return runner.epoch + 1 == runner.max_epochs

    def is_last_iter(self, runner):
        return runner.iter + 1 == runner.max_iters

    def close(self):
        """If a hook need specific operations before exit, re-write this function"""
        pass