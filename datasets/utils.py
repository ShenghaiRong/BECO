import torch
from torch import Tensor
import torch.distributed as dist
import seaborn as sns
import matplotlib.pyplot as plt

from utils.distributed import get_device


class LabelCounter:
    """
    Give label analitic info of a dataset

    Parameters:
        mode: "img_level": count for the img number containing certain class
          "pxl_level": count for pixels belong to certain class

        label: B * H * W of prediction label
    """
    
    def __init__(self, class_num: int, mode: str) -> None:
        self.class_num = class_num
        self.mode = mode
        self.device = get_device()
        self.counter = torch.zeros(class_num, dtype=torch.int, device=self.device)

    def update(self, label: Tensor):
        amount = torch.zeros(self.class_num, dtype=torch.int, device=self.device)
        if self.mode == "img_level":
            for cls in range(self.class_num):
                label_sum = torch.sum((label == cls).int(), dim=(-1, -2))
                amount[cls] = torch.sum((label_sum > 0).int())

        elif self.mode == "pxl_level":
            for cls in range(self.class_num):
                amount[cls] = torch.sum((label == cls).int())

        self.counter += amount

    def get_result(self):
        return self.counter.tolist()

    def get_figure(self):
        x = [i for i in range(self.class_num)]
        y = self.counter.tolist()
        f, ax = plt.subplots()
        ax = sns.barplot(x=x, y=y, color="0.5")
        return f

    def all_reduce(self):
        dist.all_reduce(self.counter)
