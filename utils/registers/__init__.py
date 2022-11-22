from .register import Register
from .cust_reg import *

"""
All the registers should be defined here
"""

DATASETS_REG = Register('DATASETS')

HOOKS_REG = Register('HOOKS')
HOOKS_REG.add_children(Register('VISUAL'))

MODELS_REG = Register('MODELS')

MODULES_REG = Register('MODULES')
MODULES_REG.add_children(Register('BACKBONES'))
MODULES_REG.add_children(Register('DECODERS'))
MODULES_REG.add_children(Register('MEMORY'))

NETWORKS_REG = Register('NETWORKS')

RUNNERS_REG = Register('RUNNERS')

SCHEDULER_REG = Register('SCHEDULER')

OPTIM_REG = Register('OPTIMIZER')
