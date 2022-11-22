from abc import abstractclassmethod
from typing import Dict, Any, Callable

import torch.utils.data as data


class BaseDataset(data.Dataset):
    """
    This class should not be used directly. It should only be inherited

    Parameters:
    root: path to dataset
    mode: load train set or val set
    transform: torchvision.transform
    """

    def __init__(
        self,
        root: str,
        mode: str='train',
        transform: Callable=None,
    ):

        self.root = root
        self.mode = mode
        self.transform = transform

    def disable_transform(self):
        self.transform = None

    def __getitem__(self, index) -> Dict[str, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple(image, target):
                type is torch Tensor if using transform
                type is PIL Image if not using transform

        """
        image = None
        target = None

        if self.transform is not None:
            image, target = self.transform(image, target)
        return {'img': image, 'target': target}

    @abstractclassmethod
    def __len__(self):
        pass

    @abstractclassmethod
    def get_info(self) -> str:
        pass