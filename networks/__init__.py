from typing import List
from copy import deepcopy

from utils.distributed import get_device
from utils.modules import freeze, unfreeze
from utils.misc import rgetattr

from . import deeplabv2
from . import deeplabv3
from . import deeplabv3plus
from . import resnet
from . import segformer
