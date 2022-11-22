from typing import Dict, Union, Any
from collections import OrderedDict, deque
from copy import deepcopy

import numpy as np
from torch import Tensor


class RuntimeBuffer:
    """
    A class to store runtime information. Each entry in the dict is an array.
    The format within an array should be either `Tensor` or `Dict[str, Tensor]`
    """
    def __init__(self, max_length: int = 0) -> None:
        self.buffer = OrderedDict()
        self.max_length = max_length

    def update(self, key:str, content: Any):
        """
        update the buffer if key already exists, otherwize addtionally create a new key
        """
        if key not in self.buffer:
            self.buffer[key] = deque()
        self.buffer[key].append(content)
        self._remove_oldest(key)

    def remove(self, key: str):
        if key in self.buffer:
            self.buffer.pop(key)

    def reset(self, key: str):
        if key in self.buffer:
            self.buffer[key] = deque()

    def reset_all(self):
        self.buffer = OrderedDict()

    def get(self, key: str):
        if key in self.buffer:
            return self.buffer[key]
        else:
            return None

    def get_all(self):
        return deepcopy(self.buffer)

    def get_average(self, key: str) -> Union[Dict, float]:
        """
        Return averaged contents in buffer
        """
        assert key in self.buffer
        assert len(self.buffer[key]) > 0

        if isinstance(self.buffer[key][0], dict):
            avg_content = dict()
            for k in self.buffer[key][0].keys():
                avg_content[k] = np.mean([_[k].item() for _ in self.buffer[key]]).astype(float)
        elif isinstance(self.buffer[key][0], float):
            avg_content = np.mean(self.buffer[key].item()).astype(float)
        
        return avg_content

    def _remove_oldest(self, key: str):
        if self.max_length > 0:
            while len(self.buffer[key]) > self.max_length:
                self.buffer[key].popleft()


class SimpleBuffer:

    def __init__(self):
        self.buffer = OrderedDict()

    def update_from_dict(self, content_dict):
        for k, v in content_dict.items():
            self.update(k, v)

    def update(self, key:str, content: Any):
        self.buffer[key] = content

    def get(self, key: str):
        if key in self.buffer:
            return self.buffer[key]
        else:
            return None

    def reset(self, key: str):
        if key in self.buffer:
            self.buffer.pop(key)

    def reset_all(self):
        self.buffer = OrderedDict()

    def state_dict(self):
        return self.buffer

    def load_state_dict(self, state_dict: Dict):
        for k, v in state_dict.items():
            self.buffer[k] = v
        

    