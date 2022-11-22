import torch.nn as nn

def freeze(network: nn.Module) -> None:
    """
    Freeze all params in a model
    """
    for param in network.parameters():
        param.requires_grad = False
    network.eval()

def unfreeze(network: nn.Module) -> None:
    """
    Unfreeze all params in a model
    """
    for param in network.parameters():
        param.requires_grad = True
    network.train()

def init_weight(module: nn.Module, a=0, mode='fan_in', nonlinearity='relu'):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)