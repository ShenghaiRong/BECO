from typing import Dict

from utils.registers import MODULES_REG
from . import aspp
from . import sep_aspp
from . import segformer_head


def get_decoder(config: Dict):
    decoder_obj = MODULES_REG.DECODERS.get(config.type)
    return decoder_obj(**config.settings)