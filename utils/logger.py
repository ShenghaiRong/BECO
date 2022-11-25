import logging

import utils


class R0Logger:
    """
    A wrapper of logging.Logger for logging only on rank 0
    """
    def __init__(self, name: str) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.rank, _ = utils.distributed.get_dist_info()

    def error(self, msg: str) -> None:
        if self.rank == 0:
            self.logger.error(msg)

    def warning(self, msg: str) -> None:
        if self.rank == 0:
            self.logger.warning(msg)

    def info(self, msg: str) -> None:
        if self.rank == 0:
            self.logger.info(msg)

    def debug(self, msg: str) -> None:
        if self.rank == 0:
            self.logger.debug(msg)

    def setLevel(self, level) -> None:
        if self.rank == 0:
            self.logger.setLevel(level)


class DummyLogger:
    """Used for processes with rank > 0 to ignore logging behavior"""
    def write(self, msg: str):
        pass
